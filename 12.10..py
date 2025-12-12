import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
     device = torch.device("mps")
     print("Using device: MPS (Apple Silicon GPU)")
else: device = torch.device("cpu")
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
# 2) Surrogate LIF ODE + cumulative spike series
# ============================================================
class LIF(nn.Module):
    def __init__(self, tau=20.0, threshold=1.0):
        super().__init__()
        self.tau = tau
        self.th = threshold

    def forward(self, I):
        """
        I: [B, L]
        Returns:
          spikes       : [B, L] STE spikes (for raster/plots)
          spike_series : [B, L] cumulative sum s_num over time
          hard_first   : [B]    first hard spike index (L if none) for attribution/debug
        """
        B, L = I.shape

        # keep v as [B, 1] for clean broadcasting
        v = torch.zeros(B, 1, device=I.device, dtype=I.dtype)

        spikes = torch.zeros(B, L, device=I.device, dtype=I.dtype)
        spike_series = torch.zeros(B, L, device=I.device, dtype=I.dtype)

        # s_num: [B, 1] cumulative sum of s
        s_num = torch.zeros(B, 1, device=I.device, dtype=I.dtype)

        # for attribution/debug: first hard spike time
        hard_first = torch.full((B,), L, device=I.device, dtype=torch.long)
        has_spiked = torch.zeros(B, device=I.device, dtype=torch.bool)

        k = 10.0

        for t in range(L):
            dv = -v / self.tau + I[:, t].unsqueeze(1)  # [B, 1]
            v = v + dv

            s_soft = torch.sigmoid(k * (v - self.th))      # [B, 1]
            s_hard = (v >= self.th).to(I.dtype)            # [B, 1]
            s = s_hard.detach() - s_soft.detach() + s_soft # STE, [B, 1]

            # spike raster (as before)
            spikes[:, t] = s.squeeze(1)

            # accumulate s_num and write cumulative series
            s_num = s_num + s
            spike_series[:, t] = s_num.squeeze(1)

            # record first hard spike time (not used in loss; fine to keep it hard)
            newly = (~has_spiked) & (s_hard.squeeze(1) > 0)
            if newly.any():
                hard_first[newly] = t
                has_spiked[newly] = True

            # reset potential after spike (stop-grad on spike)
            v = v * (1 - s.detach())

        return spikes, spike_series, hard_first


# ============================================================
# 3) Deep dendrites with suppressive winner-dominant competition
# ============================================================
class DeepDendrites(nn.Module):
    def __init__(self, T, K1=16, K2=8, temperature=0.1, inhibition_strength=1.0):
        super().__init__()
        self.W1 = nn.Linear(T, K1)
        self.W2 = nn.Linear(K1, K2)
        self.temperature = temperature
        self.inhibition_strength = inhibition_strength

    def forward(self, H):
        d1 = torch.relu(self.W1(H))   # [B, L, K1]
        d2 = torch.relu(self.W2(d1))  # [B, L, K2]

        logits = d2 / self.temperature
        soft_w = torch.softmax(logits, dim=-1)  # [B, L, K2]

        mean_w = soft_w.mean(dim=-1, keepdim=True)
        inhibited = soft_w - self.inhibition_strength * mean_w
        inhibited = torch.relu(inhibited)

        w_sur = inhibited / (inhibited.sum(dim=-1, keepdim=True) + 1e-6)

        idx = d2.argmax(dim=-1)  # [B, L]
        hard_w = F.one_hot(idx, d2.shape[-1]).float()

        w = hard_w.detach() - w_sur.detach() + w_sur  # STE weights

        soma_in = (w * d2).sum(dim=-1)  # [B, L]
        vals = (hard_w * d2).sum(dim=-1)

        return d2, idx, vals, w, soma_in


# ============================================================
# 4) Full self-supervised latency predictor (now series-based)
# ============================================================
class LatencyPredictor(nn.Module):
    def __init__(self, T, K1=16, K2=8):
        super().__init__()
        self.encA = DeepDendrites(T, K1, K2)
        self.encB = DeepDendrites(T, K1, K2)
        self.lifA = LIF()
        self.lifB = LIF()
        self.T = T

    def forward(self, xa, xb):
        Ha = hankel_backward(xa, self.T)
        Hb = hankel_backward(xb, self.T)

        da, ida, vala, wa, Ia = self.encA(Ha)
        db, idb, valb, wb, Ib = self.encB(Hb)

        sa, series_a, first_a = self.lifA(Ia)
        sb, series_b, first_b = self.lifB(Ib)

        return (da, ida, vala, wa, sa, series_a, first_a), \
               (db, idb, valb, wb, sb, series_b, first_b)


# ============================================================
# 5) Series-L1 loss (fully differentiable)
# ============================================================
def series_l1_loss(series_a, series_b):
    # series_*: [B, L]
    return torch.abs(series_a - series_b).mean()


# ============================================================
# 6) τ-weighted dendritic attribution (C)
# ============================================================
def tau_weighted_dendritic_attribution(d2, w, hard_latencies, tau, window_factor=3.0):
    B, L, K2 = d2.shape
    device = d2.device

    I_d = w * d2  # [B, L, K2]
    attribution = torch.zeros(B, K2, device=device)

    window = int(window_factor * tau)

    for b in range(B):
        t_spike = int(hard_latencies[b].item())
        if t_spike >= L:
            continue

        t0 = max(0, t_spike - window)
        dt = torch.arange(t0, t_spike + 1, device=device)
        decay = torch.exp(-(t_spike - dt) / tau).unsqueeze(-1)

        I_win = I_d[b, t0:t_spike + 1, :]
        contrib = (I_win * decay).sum(dim=0)

        total = contrib.sum() + 1e-6
        attribution[b] = contrib / total

    return attribution


# ============================================================
# 7) Visualization utilities (unchanged)
# ============================================================
def plot_spike_rasters_with_winners(time, spikes_a, winner_idx_a,
                                    spikes_b, winner_idx_b, threshold=0.5):
    L = spikes_a.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.plot(time, spikes_a, color='red', lw=0.2, alpha=0.7)
    ax.set_ylabel("Spike strength", color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_title("Channel A (Noisy)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    spike_mask_a = spikes_a > threshold
    winner_on_spike_a = torch.where(spike_mask_a, winner_idx_a.float(), torch.nan)
    ax2.scatter(time[spike_mask_a], winner_on_spike_a[spike_mask_a],
                color='blue', s=50, alpha=0.8)
    ax2.set_ylabel("Winning dendrite", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(-0.5, winner_idx_a.max().item() + 1)

    ax = axes[1]
    ax.plot(time, spikes_b, color='orange', lw=0.2, alpha=0.7)
    ax.set_ylabel("Spike strength", color='orange')
    ax.tick_params(axis='y', labelcolor='orange')
    ax.set_title("Channel B (Clean)")
    ax.set_xlabel("Real time")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    spike_mask_b = spikes_b > threshold
    winner_on_spike_b = torch.where(spike_mask_b, winner_idx_b.float(), torch.nan)
    ax2.scatter(time[spike_mask_b], winner_on_spike_b[spike_mask_b],
                color='green', s=50, alpha=0.8)
    ax2.set_ylabel("Winning dendrite", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(-0.5, winner_idx_b.max().item() + 1)

    plt.tight_layout()
    plt.show()


def plot_dendrite_io(time, x, dendritic_outputs):
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
    (da, ida, vala, wa, sa, series_a, first_a), \
    (db, idb, valb, wb, sb, series_b, first_b) = model(noisy, clean)

    # differentiable series L1 loss
    loss = series_l1_loss(series_a, series_b)

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(float(loss))

    if step % 200 == 0:
        print(step, float(loss))

# ============================================================
# 10) Visualization + example attribution
# ============================================================
DO_PLOT = True
if DO_PLOT:
    plt.figure(figsize=(10, 4))
    plt.plot(losses, lw=1)
    plt.xlabel("Training step")
    plt.ylabel("Loss (series L1 mismatch)")
    plt.title("Training Progress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

T_window = 20
L_vis = da.shape[1]
time = t[0, T_window - 1:T_window - 1 + L_vis].detach().cpu()

if DO_PLOT:
    plot_spike_rasters_with_winners(time,
                                    sa[0].detach().cpu(), ida[0].detach().cpu(),
                                    sb[0].detach().cpu(), idb[0].detach().cpu())

    plot_dendrite_io(time,
                     noisy[0, T_window - 1:T_window - 1 + L_vis].detach().cpu(),
                     da[0].detach().cpu())

# Attribution uses hard first-spike indices returned by LIF
attrA = tau_weighted_dendritic_attribution(da, wa, first_a, model.lifA.tau)
attrB = tau_weighted_dendritic_attribution(db, wb, first_b, model.lifB.tau)

print("τ-weighted dendritic attribution (channel A):", attrA[0].detach().cpu().numpy())
print("τ-weighted dendritic attribution (channel B):", attrB[0].detach().cpu().numpy())
