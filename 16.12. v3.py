import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# NEW: for colored line segments
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using device: MPS (Apple Silicon GPU)")
device = torch.device("cpu")
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
# 2) Surrogate LIF (NO refractory) with adaptation + hyperpolarization reset
# ============================================================
class LIF(nn.Module):
    def __init__(
        self,
        tau=20.0,
        threshold=1.0,
        tau_adapt=100.0,     # slow adaptation time constant
        beta_adapt=1.5,      # threshold increase strength
        v_reset=-0.5         # hyperpolarizing reset value
    ):
        super().__init__()
        self.tau = tau
        self.th = threshold
        self.tau_adapt = tau_adapt
        self.beta_adapt = beta_adapt
        self.v_reset = v_reset

    def forward(self, I):
        """
        I: [B, L]
        Returns:
          spikes       : [B, L] STE spikes (for raster/plots)
          spike_series : [B, L] cumulative sum s_num over time
          hard_first   : [B]    first hard spike index (L if none) for attribution/debug
        """
        B, L = I.shape
        device = I.device
        dtype = I.dtype

        v = torch.zeros(B, 1, device=device, dtype=dtype)
        a = torch.zeros(B, 1, device=device, dtype=dtype)   # adaptation variable

        spikes = torch.zeros(B, L, device=device, dtype=dtype)
        spike_series = torch.zeros(B, L, device=device, dtype=dtype)

        s_num = torch.zeros(B, 1, device=device, dtype=dtype)

        hard_first = torch.full((B,), L, device=device, dtype=torch.long)
        has_spiked = torch.zeros(B, device=device, dtype=torch.bool)

        k = 10.0  # surrogate slope

        for t in range(L):
            # effective threshold with adaptation
            th_eff = self.th + self.beta_adapt * a

            # integrate (NO refractory masking)
            dv = (-v / self.tau + I[:, t].unsqueeze(1))
            v = v + dv

            # surrogate spike
            s_soft = torch.sigmoid(k * (v - th_eff))
            s_hard = (v >= th_eff).to(dtype)

            s = s_hard.detach() - s_soft.detach() + s_soft  # STE

            spikes[:, t] = s.squeeze(1)

            # cumulative spike series
            s_num = s_num + s
            spike_series[:, t] = s_num.squeeze(1)

            # first hard spike time (debug/attr)
            newly = (~has_spiked) & (s_hard.squeeze(1) > 0)
            if newly.any():
                hard_first[newly] = t
                has_spiked[newly] = True

            # reset with hyperpolarization (stop-grad on spike)
            v = v * (1 - s.detach()) + self.v_reset * s.detach()

            # adaptation dynamics
            da = -a / self.tau_adapt + s
            a = a + da

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
# 4) Full self-supervised latency predictor (series-based)
# ============================================================
class LatencyPredictor(nn.Module):
    def __init__(self, T, K1=16, K2=8):
        super().__init__()
        self.encA = DeepDendrites(T, K1, K2)
        self.encB = DeepDendrites(T, K1, K2)

        # LIF WITHOUT refractory
        self.lifA = LIF(tau=20.0, threshold=1.5, tau_adapt=100.0, beta_adapt=1.5,
                        v_reset=-0.5)
        self.lifB = LIF(tau=20.0, threshold=1.5, tau_adapt=100.0, beta_adapt=1.5,
                        v_reset=-0.5)

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
# 7) Visualization utilities
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
# 7b) Spike-Centric "Causal Stack" plot
#     UPDATED: no heatmap panel, credit is encoded ON the curves
# ============================================================
def _rising_edges(spikes_1d, thr=0.5, refractory=3):
    """
    spikes_1d: [L] soft spikes
    Returns indices of rising edges (events). Optional simple index-space refractory.
    """
    s = spikes_1d.detach().cpu()
    above = (s > thr).to(torch.int32)
    rise = torch.where((above[1:] == 1) & (above[:-1] == 0))[0] + 1
    if rise.numel() == 0:
        return rise

    keep = [int(rise[0].item())]
    last = keep[0]
    for idx in rise[1:]:
        ti = int(idx.item())
        if ti - last >= refractory:
            keep.append(ti)
            last = ti
    return torch.tensor(keep, dtype=torch.long)


def _colored_line(ax, x, y, c, lw=2.0, alpha=1.0):
    """
    Plot a line y(x) with color mapped from c at each x (segment-wise).
    x, y, c: 1D tensors/arrays, same length
    """
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
    if isinstance(c, torch.Tensor): c = c.detach().cpu().numpy()

    # line segments between consecutive points
    pts = torch.tensor(list(zip(x, y))).numpy()
    segs = torch.stack([torch.tensor(pts[:-1]), torch.tensor(pts[1:])], dim=1).numpy()

    lc = LineCollection(segs, array=c[:-1], linewidths=lw, alpha=alpha)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


def plot_causal_stack_per_spike(
    time, x, d2, w, spikes, winner_idx=None,
    tau=20.0, spike_thr=0.5, window_factor=3.0, max_spikes=6,
    event_refractory=3,
    show_colorbar=True
):
    """
    For each spike event (rising edge), plot:
      A) x(t) around spike
      B) dendrite contributions I_k(t)=w*d2  (ONLY winner + runner-up),
         colored along the curve by causal credit C_k(t)=I_k(t)*exp(-(t_s-t)/tau)
    """
    time_t = time.detach().cpu() if isinstance(time, torch.Tensor) else torch.tensor(time)
    x_t    = x.detach().cpu()    if isinstance(x, torch.Tensor) else torch.tensor(x)
    d2_t   = d2.detach().cpu()
    w_t    = w.detach().cpu()
    sp_t   = spikes.detach().cpu()
    win_t  = winner_idx.detach().cpu() if winner_idx is not None else None

    L, K2 = d2_t.shape
    tau_f = float(tau)
    win_len = int(window_factor * tau_f)

    I_d = w_t * d2_t  # [L, K2]

    events = _rising_edges(sp_t, thr=spike_thr, refractory=event_refractory)
    if events.numel() == 0:
        print("No spike events (rising edges) found above threshold.")
        return
    events = events[:max_spikes]
    n = int(events.numel())

    # normalize colors across all selected spikes for comparability
    all_credits = []
    for t_s in events.tolist():
        t0 = max(0, t_s - win_len)
        idx = torch.arange(t0, t_s + 1)
        decay = torch.exp(-(t_s - idx).float() / tau_f)  # [win]
        C_all = I_d[idx, :] * decay.unsqueeze(1)         # [win, K2]
        all_credits.append(C_all)
    C_cat = torch.cat(all_credits, dim=0)
    vmin = float(torch.min(C_cat).item())
    vmax = float(torch.max(C_cat).item()) + 1e-12
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(13, 2.6 * n), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2 * n, ncols=1)

    first_lc = None

    for i, t_s in enumerate(events.tolist()):
        t0 = max(0, t_s - win_len)
        idx = torch.arange(t0, t_s + 1)

        decay = torch.exp(-(t_s - idx).float() / tau_f)  # [win]
        C = I_d[idx, :] * decay.unsqueeze(1)             # [win, K2]

        # winner at spike time
        k_star = int(win_t[t_s].item()) if win_t is not None else None

        # runner-up by total causal credit in the same window (exclude winner)
        totals = C.sum(dim=0)  # [K2]
        if k_star is not None and 0 <= k_star < K2:
            totals_ru = totals.clone()
            totals_ru[k_star] = -1e9
            k_ru = int(torch.argmax(totals_ru).item())
        else:
            k_ru = int(torch.argmax(totals).item())
            k_star = k_ru  # fallback

        # --- Track A: Input ---
        axA = fig.add_subplot(gs[2*i + 0, 0])
        axA.plot(time_t[idx], x_t[idx], lw=1.2)
        axA.axvline(time_t[t_s].item(), lw=1.0)
        axA.set_ylabel("x(t)")
        axA.grid(True, alpha=0.25)
        axA.set_title(
            f"Spike {i+1} | t_idx={t_s} | winner D{k_star} | runner-up D{k_ru} | lookback≈{win_len}≈{window_factor}τ | τ={tau_f:g}"
        )

        # --- Track B: Winner + runner-up (colored by credit) ---
        axB = fig.add_subplot(gs[2*i + 1, 0], sharex=axA)

        lc1 = _colored_line(axB, time_t[idx], I_d[idx, k_star], C[:, k_star], lw=2.6, alpha=0.95)
        lc1.set_norm(norm)

        lc2 = _colored_line(axB, time_t[idx], I_d[idx, k_ru],   C[:, k_ru],   lw=1.6, alpha=0.85)
        lc2.set_norm(norm)

        axB.axvline(time_t[t_s].item(), lw=1.0)
        axB.set_ylabel("I_k(t)")
        axB.grid(True, alpha=0.25)

        # lightweight legend (line samples)
        axB.plot([], [], lw=2.6, label=f"winner D{k_star} (colored)")
        axB.plot([], [], lw=1.6, label=f"runner-up D{k_ru} (colored)")
        axB.legend(loc="upper right", frameon=False)

        if first_lc is None:
            first_lc = lc1

        if i == n - 1:
            axB.set_xlabel("time")

    if show_colorbar and first_lc is not None:
        fig.colorbar(first_lc, ax=fig.axes, fraction=0.02, pad=0.01, label="causal credit")

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

lambda_spike = 1e-3  # spike-cost strength (try 1e-4 .. 1e-2 if needed)

for step in range(1000):
    (da, ida, vala, wa, sa, series_a, first_a), \
    (db, idb, valb, wb, sb, series_b, first_b) = model(noisy, clean)

    # differentiable series L1 loss + spike-rate regularizer
    loss = series_l1_loss(series_a, series_b) + lambda_spike * (sa.mean() + sb.mean())

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(float(loss))

    if step % 20 == 0:
        print(step, float(loss))


# ============================================================
# 10) Visualization + example attribution
# ============================================================
DO_PLOT = True
if DO_PLOT:
    plt.figure(figsize=(10, 4))
    plt.plot(losses, lw=1)
    plt.xlabel("Training step")
    plt.ylabel("Loss (series L1 + spike cost)")
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

    # UPDATED: Spike-centric causal stack (Channel A) with credit ON the curves
    plot_causal_stack_per_spike(
        time=time,
        x=noisy[0, T_window - 1:T_window - 1 + L_vis].detach().cpu(),
        d2=da[0].detach().cpu(),
        w=wa[0].detach().cpu(),
        spikes=sa[0].detach().cpu(),
        winner_idx=ida[0].detach().cpu(),
        tau=float(model.lifA.tau),
        spike_thr=0.5,
        window_factor=3.0,
        max_spikes=6,
        event_refractory=3,
        show_colorbar=True
    )

# Attribution uses hard first-spike indices returned by LIF
attrA = tau_weighted_dendritic_attribution(da, wa, first_a, model.lifA.tau)
attrB = tau_weighted_dendritic_attribution(db, wb, first_b, model.lifB.tau)

print("τ-weighted dendritic attribution (channel A):", attrA[0].detach().cpu().numpy())
print("τ-weighted dendritic attribution (channel B):", attrB[0].detach().cpu().numpy())
