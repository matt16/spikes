import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using device: MPS (Apple Silicon GPU)")
# else:
device = torch.device("cpu")

# Toggle plotting (set False to run headless/non-interactive)
DO_PLOT = False


# ============================================================
# 1) Strict backward-looking Hankel (sliding window)
# ============================================================
def hankel_backward(x, T):
    """
    x: [B, L]
    Return windows ending at each time step:
    H[b, t] = x[b, t-T : t]
    Shape: [B, L-T+1, T]
    """
    B, L = x.shape
    xs = []
    for i in range(T, L + 1):
        xs.append(x[:, i - T:i])
    return torch.stack(xs, dim=1)  # [B, L-T+1, T]


# ============================================================
# 1a) Helper
# ============================================================
# def soft_lat(spikes, time_indices):
#     """
#     spikes: [B, L] with surrogate sigmoid spikes (0 to 1)
#     time_indices: [B, L] with time values for each position
#     Returns: [B, L] soft latency values (differentiable time-instants)

#     Uses spikes as soft weights to create a smooth version of torch.where,
#     mapping each position to its estimated time index.
#     """
#     B, L = spikes.shape

#     # Normalize spikes to weights per position (soft version of "is there a spike here?")
#     # Use sigmoid or normalize to [0, 1]
#     soft_weights = torch.sigmoid(10 * (spikes - 0.5))  # Sharper soft weights

#     # For each spike position, the soft latency is the time index weighted by spike confidence
#     soft_latencies = soft_weights * time_indices  # [B, L]

#     return soft_latencies


# ============================================================
# 2) Surrogate LIF ODE + latency extraction
# ============================================================
class LIF(nn.Module):
    def __init__(self, tau=20.0, threshold=1.0):
        super().__init__()
        self.tau = tau
        self.th = threshold

    def forward(self, I):
        """
        I: [B, L]
        Euler ODE: dv = dt*(-v/tau + I)
        Surrogate spike = sigmoid(k*(v-th))
        Reset v ← v*(1 - stopgrad(spike)).
        Returns spike raster and latency of first spike.
        """
        B, L = I.shape
        v = torch.zeros(B, device=I.device)
        spikes = torch.zeros(B, L, device=I.device)
        hard_spikes = torch.zeros(B, L, device=I.device)  # binary spike raster
        k = 10.0

        for t in range(L):
            dv = -v / self.tau + I[:, t]
            v = v + dv

            # surrogate spike (soft) + hard spike with STE
            s_soft = torch.sigmoid(k * (v - self.th))  # surrogate
            s_hard = (v >= self.th).float()  # hard spike (binary)
            s = s_hard.detach() - s_soft.detach() + s_soft  # STE

            # hard_spikes[:, t] = s_hard
            spikes[:, t] = s

            # reset potential after spike (stop-grad on spike)
            v = v * (1 - s.detach())

        # Extract spike latencies as vectors from differentiable spikes
        time_indices = torch.arange(L, device=I.device, dtype=torch.float32)
        latencies = []

        for b in range(B):
            # Find spike times using surrogate spikes (differentiable)
            # Threshold at 0.5 for spike detection
            spike_mask = spikes[b] > 0.5
            spike_times = torch.where(spike_mask)[0].float()  # indices where spike occurred

            if len(spike_times) > 0:
                # Extract latencies at spike positions: spike value * time index
                spike_lats = spikes[b][spike_times.long()] * spike_times
                latencies.append(spike_lats)
            else:
                # No spikes in this sample
                latencies.append(torch.tensor([], device=I.device, dtype=torch.float32))

        return spikes, latencies


# ============================================================
# 3) Deep dendrites with suppressive winner-dominant competition
# ============================================================
class DeepDendrites(nn.Module):
    """
    Dendritic competition with suppressive, winner-dominant selection.

    - Soft competition (softmax with low temperature)
    - Lateral inhibition (subtract mean, clamp at 0, renormalise)
    - STE to keep a hard winner index for interpretability.
    """

    def __init__(self, T, K1=16, K2=8, temperature=0.1, inhibition_strength=1.0):
        super().__init__()
        self.W1 = nn.Linear(T, K1)
        self.W2 = nn.Linear(K1, K2)
        self.temperature = temperature
        self.inhibition_strength = inhibition_strength

    def forward(self, H):
        """
        H: [B, L, T]
        Returns:
          d2      : [B, L, K2] raw dendritic activations
          idx     : [B, L]     hard winner index (argmax over d2)
          vals    : [B, L]     winner activation value (for diagnostics)
          w       : [B, L, K2] effective dendritic weights after competition (STE)
          soma_in : [B, L]     scalar current to soma
        """
        # Dendritic preprocessing
        d1 = torch.relu(self.W1(H))  # [B, L, K1]
        d2 = torch.relu(self.W2(d1))  # [B, L, K2]

        # --- soft competition with lateral inhibition + renormalisation ---
        logits = d2 / self.temperature
        soft_w = torch.softmax(logits, dim=-1)  # [B, L, K2], sum_k soft_w = 1

        mean_w = soft_w.mean(dim=-1, keepdim=True)  # [B, L, 1]
        inhibited = soft_w - self.inhibition_strength * mean_w
        inhibited = torch.relu(inhibited)  # losers go to 0

        w_sur = inhibited / (inhibited.sum(dim=-1, keepdim=True) + 1e-6)  # renormalised

        # --- STE: hard winner for interpretability, soft for gradients ---
        idx = d2.argmax(dim=-1)  # [B, L]
        hard_w = F.one_hot(idx, d2.shape[-1]).float()  # [B, L, K2]

        w = hard_w.detach() - w_sur.detach() + w_sur  # STE weights

        # Current into soma: weighted dendritic activations
        soma_in = (w * d2).sum(dim=-1)  # [B, L]

        # Winner value (for diagnostics)
        vals = (hard_w * d2).sum(dim=-1)  # [B, L]

        return d2, idx, vals, w, soma_in


# ============================================================
# 4) Full self-supervised latency predictor
# ============================================================
class LatencyPredictor(nn.Module):
    """
    Two channels (A,B) supervise each other:
    Loss = |latA - latB|.
    """

    def __init__(self, T, K1=16, K2=8):
        super().__init__()
        self.encA = DeepDendrites(T, K1, K2)
        self.encB = DeepDendrites(T, K1, K2)
        self.lifA = LIF()
        self.lifB = LIF()
        self.T = T

    def forward(self, xa, xb):
        Ha = hankel_backward(xa, self.T)  # [B, L', T]
        Hb = hankel_backward(xb, self.T)

        da, ida, vala, wa, Ia = self.encA(Ha)  # d2, idx, vals, w, soma_in
        db, idb, valb, wb, Ib = self.encB(Hb)

        sa, la = self.lifA(Ia)  # spikes [B,L], latencies (list of spike time vectors)
        sb, lb = self.lifB(Ib)

        return (da, ida, vala, wa, sa, la), \
            (db, idb, valb, wb, sb, lb)


# ============================================================
# 5) Latency-matching loss
# ============================================================
def latency_loss(la, lb):
    # Build per-sample absolute differences from in-order pairs, drop excess.
    diffs = []
    for a, b in zip(la, lb):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            continue
        n = int(min(a.numel(), b.numel()))
        if n <= 0:
            continue
        diffs.append((a[:n].float().to(a.device) - b[:n].float().to(a.device)).abs())
    if not diffs:
        return torch.tensor(0.0, device=(
            la[0].device if isinstance(la, list) and len(la) > 0 and isinstance(la[0], torch.Tensor) else torch.device(
                'cpu')))
    return torch.cat(diffs).mean()


# ============================================================
# 6) τ-weighted dendritic attribution (C)
# ============================================================
def tau_weighted_dendritic_attribution(d2, w, hard_latencies, tau, window_factor=3.0):
    """
    d2           : [B, L, K2] raw dendritic activations
    w            : [B, L, K2] effective dendritic weights after competition
    hard_latencies : [B]     first-spike times (indices, from LIF.hard_latency)
    tau          : float     LIF time constant (in same time units as steps)
    window_factor: float     how many taus to look back (e.g., 3 → ~95% of mass)

    Returns:
      attribution: [B, K2]   τ-weighted contribution per dendrite for the FIRST spike
                             (normalised to sum ≈ 1 per sample; zeros if no spike)
    """
    B, L, K2 = d2.shape
    device = d2.device

    I_d = w * d2  # [B, L, K2]: effective dendritic current per branch

    attribution = torch.zeros(B, K2, device=device)

    window = int(window_factor * tau)

    for b in range(B):
        t_spike = int(hard_latencies[b].item())
        if t_spike >= L:
            # no hard spike occurred → leave attribution at zeros
            continue

        t0 = max(0, t_spike - window)
        dt = torch.arange(t0, t_spike + 1, device=device)
        decay = torch.exp(-(t_spike - dt) / tau)  # [T_win]
        decay = decay.unsqueeze(-1)  # [T_win, 1]

        I_win = I_d[b, t0:t_spike + 1, :]  # [T_win, K2]
        contrib = (I_win * decay).sum(dim=0)  # [K2]

        total = contrib.sum() + 1e-6
        attribution[b] = contrib / total

    return attribution  # [B, K2]


# ============================================================
# 7) Visualization utilities
# ============================================================
def plot_spike_rasters_with_winners(time, spikes_a, winner_idx_a,
                                    spikes_b, winner_idx_b, threshold=0.5):
    """
    Plot spike rasters from both channels side-by-side with winning dendrite index overlaid.
    Only shows winner index when an actual spike occurs (spike > threshold).

    spikes_a, spikes_b: [L] spike rasters from channels A and B
    winner_idx_a, winner_idx_b: [L] winning dendrite index at each time
    threshold: spike threshold for determining if a spike occurred
    """
    L = spikes_a.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Channel A (noisy)
    ax = axes[0]
    ax.plot(time, spikes_a, color='red', lw=0.2, alpha=0.7, label='Spikes')
    ax.set_ylabel("Spike strength", color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_title("Channel A (Noisy)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    spike_mask_a = spikes_a > threshold
    winner_on_spike_a = torch.where(spike_mask_a, winner_idx_a.float(), torch.nan)
    ax2.scatter(time[spike_mask_a], winner_on_spike_a[spike_mask_a],
                color='blue', s=50, alpha=0.8, label='Winner')
    ax2.set_ylabel("Winning dendrite", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(-0.5, winner_idx_a.max().item() + 1)

    # Channel B (clean)
    ax = axes[1]
    ax.plot(time, spikes_b, color='orange', lw=0.2, alpha=0.7, label='Spikes')
    ax.set_ylabel("Spike strength", color='orange')
    ax.tick_params(axis='y', labelcolor='orange')
    ax.set_title("Channel B (Clean)")
    ax.set_xlabel("Real time")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    spike_mask_b = spikes_b > threshold
    winner_on_spike_b = torch.where(spike_mask_b, winner_idx_b.float(), torch.nan)
    ax2.scatter(time[spike_mask_b], winner_on_spike_b[spike_mask_b],
                color='green', s=50, alpha=0.8, label='Winner')
    ax2.set_ylabel("Winning dendrite", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(-0.5, winner_idx_b.max().item() + 1)

    plt.tight_layout()
    plt.show()


def plot_dendrite_io(time, x, dendritic_outputs):
    """
    Per-dendrite visualization:
    Left column: input segment
    Right column: dendritic filtered output
    """
    L, K2 = dendritic_outputs.shape
    fig, axes = plt.subplots(K2, 2, figsize=(10, 2 * K2))
    for k in range(K2):
        axes[k, 0].plot(time, x, color='red')
        axes[k, 0].set_title(f"D{k} input")

        axes[k, 1].plot(time, dendritic_outputs[:, k], color='blue')
        axes[k, 1].set_title(f"D{k} output")

    plt.tight_layout()
    plt.show()


# ============================================================
# 8) Generate synthetic self-supervised pair
# ============================================================
t = torch.linspace(0, 4 * 3.14, 400, device=device).unsqueeze(0)
clean = torch.sin(t)
noisy = clean + 0.2 * torch.randn_like(clean)

# ============================================================
# 9) Training loop (expanded + commented)
# ============================================================
model = LatencyPredictor(T=20, K1=32, K2=6).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []

for step in range(200):
    # ---- forward pass ----
    (da, ida, vala, wa, sa, la), \
        (db, idb, valb, wb, sb, lb) = model(noisy, clean)

    # ---- compute latency-based self-supervised loss ----
    loss = latency_loss(la, lb)
    # If pairing-based loss is non-differentiable (no grad), fallback to
    # a differentiable centroid/soft-spike loss computed from surrogate spikes.
    if not (isinstance(loss, torch.Tensor) and loss.requires_grad):
        eps = 1e-8
        t_idx = torch.arange(sa.shape[1], device=device, dtype=sa.dtype).unsqueeze(0)
        pa = sa / (sa.sum(dim=1, keepdim=True) + eps)
        pb = sb / (sb.sum(dim=1, keepdim=True) + eps)
        mu_a = (pa * t_idx).sum(dim=1)
        mu_b = (pb * t_idx).sum(dim=1)
        loss = (mu_a - mu_b).abs().mean()

    # ---- optimize ----
    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(float(loss))

    if step % 200 == 0:
        print(step, float(loss))

# ============================================================
# 10) Visualization + example attribution
# ============================================================
# Training progress plot
if DO_PLOT:
    plt.figure(figsize=(10, 4))
    plt.plot(losses, lw=1)
    plt.xlabel("Training step")
    plt.ylabel("Loss (latency mismatch)")
    plt.title("Training Progress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example after training:
T_window = 20  # matches T in LatencyPredictor
L_vis = da.shape[1]  # segment length after Hankel
time = t[0, T_window - 1:T_window - 1 + L_vis].detach().cpu()  # match Hankel-aligned time window

# Plot spike rasters from both channels with winning dendrite overlaid (optional)
if DO_PLOT:
    plot_spike_rasters_with_winners(time,
                                    sa[0].detach().cpu(), ida[0].detach().cpu(),
                                    sb[0].detach().cpu(), idb[0].detach().cpu())

    # Per-dendrite outputs for channel A (noisy)
    plot_dendrite_io(time,
                     noisy[0, T_window - 1:T_window - 1 + L_vis].detach().cpu(),
                     da[0].detach().cpu())

# --- τ-weighted dendritic attribution for FIRST spike in each channel ---
# Extract first spike index from latency lists
la_first = torch.zeros(len(la), dtype=torch.long, device=la[0].device if len(la) > 0 else torch.device('cpu'))
lb_first = torch.zeros(len(lb), dtype=torch.long, device=lb[0].device if len(lb) > 0 else torch.device('cpu'))
for b in range(len(la)):
    if len(la[b]) > 0:
        la_first[b] = int(la[b][0].item())
    else:
        la_first[b] = da.shape[1]  # L dimension
    if len(lb[b]) > 0:
        lb_first[b] = int(lb[b][0].item())
    else:
        lb_first[b] = db.shape[1]  # L dimension

attrA = tau_weighted_dendritic_attribution(da, wa, la_first, model.lifA.tau)
attrB = tau_weighted_dendritic_attribution(db, wb, lb_first, model.lifB.tau)

print("τ-weighted dendritic attribution (channel A):", attrA[0].detach().cpu().numpy())
print("τ-weighted dendritic attribution (channel B):", attrB[0].detach().cpu().numpy())
