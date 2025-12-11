import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")

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
    for i in range(T, L+1):
        xs.append(x[:, i-T:i])
    return torch.stack(xs, dim=1)  # [B, L-T+1, T]


# ============================================================
# 1a) Helper
# ============================================================
def soft_lat(spikes):
    # spikes: [B, L] with surrogate sigmoid spikes
    L = spikes.shape[1]
    time = torch.arange(L, device=spikes.device, dtype=spikes.dtype)
    w = spikes / (spikes.sum(dim=1, keepdim=True) + 1e-6)
    return (w * time).sum(dim=1)


# ============================================================
# 2) Surrogate LIF ODE + latency extraction
#     + absolute refractory
#     + adaptive threshold (relative refractory)
# ============================================================
class LIF(nn.Module):
    def __init__(self,
                 tau=20.0,
                 threshold=1.0,
                 tau_theta=50.0,
                 theta_inc=0.5,
                 refrac_steps=5):
        """
        tau         : Membranzeitkonstante
        threshold   : Basis-Schwelle (theta_0)
        tau_theta   : Zeitkonstante, mit der die Schwelle zurück zur Basis fällt
        theta_inc   : wie stark die Schwelle nach einem Spike angehoben wird
        refrac_steps: absolute Refraktärzeit (in Zeitschritten)
        """
        super().__init__()
        self.tau = tau
        self.th = threshold
        self.tau_theta = tau_theta
        self.theta_inc = theta_inc
        self.refrac_steps = refrac_steps

    def forward(self, I):
        """
        I: [B, L]
        Euler ODE: dv = dt*(-v/tau + I)
        Surrogate spike = sigmoid(k*(v-theta_t))
        Reset v ← v*(1 - stopgrad(spike)).
        Plus:
          - absolute Refraktärzeit über ref_count
          - adaptive Schwelle theta_t
        Returns spike raster and latency of first spike.
        """
        B, L = I.shape
        device = I.device

        v = torch.zeros(B, device=device)              # Membranpotential
        theta = torch.full((B,), self.th, device=device)  # dynamische Schwelle
        ref_count = torch.zeros(B, dtype=torch.long, device=device)  # Refraktärzähler

        spikes = torch.zeros(B, L, device=device)
        hard_spikes = torch.zeros(B, L, device=device)  # binär (nur für Latenz)
        k = 10.0  # Steilheit der Surrogate-Funktion

        for t in range(L):
            # Membrandynamik immer updaten (auch in Refraktärzeit)
            dv = -v / self.tau + I[:, t]
            v = v + dv

            # Nur Neuronen mit ref_count == 0 dürfen überhaupt feuern
            can_spike = (ref_count == 0).float()

            # Abstand zur aktuellen Schwelle
            x = v - theta

            # Surrogate-Spike (weiche Version, aber nur wenn can_spike)
            s_soft = torch.sigmoid(k * x) * can_spike

            # Harte Spike-Entscheidung: nur, wenn über Schwelle und nicht refraktär
            s_hard = ((x >= 0.0) & (ref_count == 0)).float()

            # Straight-Through-Estimator
            s = s_hard.detach() - s_soft.detach() + s_soft

            hard_spikes[:, t] = s_hard
            spikes[:, t] = s

            # Absolute Refraktärzeit:
            # bei Spike -> Zähler auf refrac_steps setzen
            # sonst: Zähler runterzählen, aber nicht unter 0
            new_ref = torch.where(
                s_hard > 0,
                torch.full_like(ref_count, self.refrac_steps),
                torch.clamp(ref_count - 1, min=0)
            )
            ref_count = new_ref

            # Reset des Potentials nach Spike (surrogate-basiert)
            v = v * (1 - s.detach())

            # Adaptive Schwelle:
            # 1) Drift zurück zur Basis-Schwelle self.th
            theta = theta + (-(theta - self.th) / self.tau_theta)
            # 2) Sprung nach oben bei hartem Spike
            theta = theta + self.theta_inc * s_hard

        # Latenz: erste harte Spike-Zeit pro Sample
        first_spikes = (hard_spikes > 0.5).int()
        first_spike_times = torch.full((B,), L, dtype=torch.long, device=device)
        for b in range(B):
            spike_times = torch.where(first_spikes[b] > 0)[0]
            if len(spike_times) > 0:
                first_spike_times[b] = spike_times[0]

        hard_latency = first_spike_times
        soft_latency = soft_lat(spikes)   # float, differentiable

        return spikes, hard_latency, soft_latency


# ============================================================
# 3) Deep dendrites with suppressive winner-dominant competition
# ============================================================
class DeepDendrites(nn.Module):
    """
    Dendritic competition with suppressive, winner-dominant selection.

    - Soft competition (softmax with low temperature)
    - Lateral inhibition (subtract mean, clamp at 0, renormalise)
    - Multiple dendrites can contribute in forward; argmax only for diagnostics.
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
          idx     : [B, L]     hard winner index (argmax over d2) for diagnostics
          vals    : [B, L]     winner activation value
          w       : [B, L, K2] soft competitive weights
          soma_in : [B, L]     scalar current to soma
        """
        d1 = torch.relu(self.W1(H))     # [B, L, K1]
        d2 = torch.relu(self.W2(d1))    # [B, L, K2]

        # soft competition + laterale Inhibition
        logits = d2 / self.temperature
        soft_w = torch.softmax(logits, dim=-1)          # [B, L, K2]

        mean_w = soft_w.mean(dim=-1, keepdim=True)      # [B, L, 1]
        inhibited = soft_w - self.inhibition_strength * mean_w
        inhibited = torch.relu(inhibited)               # Verlierer → 0

        w = inhibited / (inhibited.sum(dim=-1, keepdim=True) + 1e-6)  # [B, L, K2]

        soma_in = (w * d2).sum(dim=-1)                  # [B, L]

        vals, idx = d2.max(dim=-1)                      # nur für Diagnose

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

        da, ida, vala, wa, Ia = self.encA(Ha)   # d2, idx, vals, w, soma_in
        db, idb, valb, wb, Ib = self.encB(Hb)

        sa, la_hard, la_soft = self.lifA(Ia)    # spikes, first hard latency, soft latency
        sb, lb_hard, lb_soft = self.lifB(Ib)

        return (da, ida, vala, wa, Ia, sa, la_hard, la_soft), \
               (db, idb, valb, wb, Ib, sb, lb_hard, lb_soft)


# ============================================================
# 5) Latency-matching loss
# ============================================================
def latency_loss(la, lb):
    return torch.mean(torch.abs(la.float() - lb.float()))


# ============================================================
# 6) τ-weighted dendritic attribution (C)
# ============================================================
def tau_weighted_dendritic_attribution(d2, w, hard_latencies, tau, window_factor=3.0):
    """
    d2           : [B, L, K2] raw dendritic activations
    w            : [B, L, K2] competitive weights
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
            # kein harter Spike → Attribution bleibt 0
            continue

        t0 = max(0, t_spike - window)
        dt = torch.arange(t0, t_spike + 1, device=device)
        decay = torch.exp(-(t_spike - dt) / tau)   # [T_win]
        decay = decay.unsqueeze(-1)                # [T_win, 1]

        I_win = I_d[b, t0:t_spike + 1, :]          # [T_win, K2]
        contrib = (I_win * decay).sum(dim=0)       # [K2]

        total = contrib.sum() + 1e-6
        attribution[b] = contrib / total

    return attribution  # [B, K2]


# ============================================================
# 7) Visualization utilities
# ============================================================
def plot_soma_with_winners(time,
                           soma_a, winner_idx_a,
                           soma_b, winner_idx_b,
                           frac_threshold=0.5):
    """
    Plot soma current from both channels with winning dendrite index overlaid.
    """
    soma_a = soma_a.detach().cpu()
    soma_b = soma_b.detach().cpu()
    winner_idx_a = winner_idx_a.detach().cpu()
    winner_idx_b = winner_idx_b.detach().cpu()
    time = time.detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Channel A (noisy)
    ax = axes[0]
    ax.plot(time, soma_a, lw=1.0, alpha=0.8)
    ax.set_ylabel("Soma current (A)")
    ax.set_title("Channel A (Noisy)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    if soma_a.numel() > 0:
        thr_a = frac_threshold * float(soma_a.max().item())
    else:
        thr_a = 0.0
    mask_a = soma_a > thr_a
    winner_on_a = torch.where(mask_a, winner_idx_a.float(), torch.nan)
    ax2.scatter(time[mask_a], winner_on_a[mask_a],
                s=40, alpha=0.9)
    ax2.set_ylabel("Winning dendrite (A)")
    ax2.set_ylim(-0.5, winner_idx_a.max().item() + 1)

    # Channel B (clean)
    ax = axes[1]
    ax.plot(time, soma_b, lw=1.0, alpha=0.8)
    ax.set_ylabel("Soma current (B)")
    ax.set_title("Channel B (Clean)")
    ax.set_xlabel("Real time")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    if soma_b.numel() > 0:
        thr_b = frac_threshold * float(soma_b.max().item())
    else:
        thr_b = 0.0
    mask_b = soma_b > thr_b
    winner_on_b = torch.where(mask_b, winner_idx_b.float(), torch.nan)
    ax2.scatter(time[mask_b], winner_on_b[mask_b],
                s=40, alpha=0.9)
    ax2.set_ylabel("Winning dendrite (B)")
    ax2.set_ylim(-0.5, winner_idx_b.max().item() + 1)

    plt.tight_layout()
    plt.show()


def plot_dendrite_io(time, x, dendritic_outputs):
    """
    Per-dendrite visualization:
    Left column: input segment
    Right column: dendritic filtered output
    """
    time = time.detach().cpu()
    x = x.detach().cpu()
    dendritic_outputs = dendritic_outputs.detach().cpu()

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
# 9) Training loop
# ============================================================
model = LatencyPredictor(T=20, K1=32, K2=6).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []

for step in range(2000):
    (da, ida, vala, wa, Ia, sa, la_hard, la_soft), \
    (db, idb, valb, wb, Ib, sb, lb_hard, lb_soft) = model(noisy, clean)

    loss = latency_loss(la_soft, lb_soft)

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(float(loss))

    if step % 200 == 0:
        print(step, float(loss))

# ============================================================
# 10) Visualisation + example attribution
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(losses, lw=1)
plt.xlabel("Training step")
plt.ylabel("Loss (latency mismatch)")
plt.title("Training Progress")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

T_window = model.T
L_vis = da.shape[1]
time_vis = t[0, T_window - 1:T_window - 1 + L_vis].detach().cpu()

plot_soma_with_winners(
    time_vis,
    Ia[0, :L_vis],
    ida[0, :L_vis],
    Ib[0, :L_vis],
    idb[0, :L_vis],
    frac_threshold=0.5
)

plot_dendrite_io(
    time_vis,
    noisy[0, T_window - 1:T_window - 1 + L_vis],
    da[0, :L_vis]
)

attrA = tau_weighted_dendritic_attribution(da, wa, la_hard, model.lifA.tau)
attrB = tau_weighted_dendritic_attribution(db, wb, lb_hard, model.lifB.tau)

print("τ-weighted dendritic attribution (channel A):", attrA[0].detach().cpu().numpy())
print("τ-weighted dendritic attribution (channel B):", attrB[0].detach().cpu().numpy())
