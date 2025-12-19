import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

device = torch.device("cpu")
DO_PLOT = True


# ============================================================
# 1) Causal Conv Filter Bank (replaces Hankel+MLP)
# ============================================================
class ConvFilterBank(nn.Module):
    """
    Runs a bank of causal 1D conv filters with stride=1.
    Each filter output at time t depends on x[t-(k-1) ... t] (last value is always x[t]).
    Variable kernel sizes are supported by grouping filters per kernel size.

    Output:
      d2: [B, L_eff, K]  (ReLU activations, cropped to valid region based on max kernel)
      warmup: int = (max_kernel_size - 1)
    """
    def __init__(self, kernel_sizes, bias=True):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.kmax = max(self.kernel_sizes)

        # group by kernel size
        size_to_count = {}
        for k in self.kernel_sizes:
            size_to_count[k] = size_to_count.get(k, 0) + 1

        # deterministic order for concat
        self._unique_sizes = sorted(size_to_count.keys())
        self._counts = [size_to_count[k] for k in self._unique_sizes]

        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=size_to_count[k], kernel_size=k, stride=1, bias=bias)
            for k in self._unique_sizes
        ])

        self.K = sum(self._counts)

    def forward(self, x):
        # x: [B, L]
        B, L = x.shape
        x1 = x.unsqueeze(1)  # [B, 1, L]

        outs = []
        for conv, k in zip(self.branches, self._unique_sizes):
            # causal left padding: pad (k-1) zeros on the left
            xpad = F.pad(x1, (k - 1, 0))
            y = conv(xpad)         # [B, Ck, L]
            y = F.relu(y)          # dendritic values
            outs.append(y)

        # [B, K, L] -> [B, L, K]
        y_all = torch.cat(outs, dim=1)
        d2 = y_all.transpose(1, 2)

        # crop warmup so all filters are comparable (full context for max kernel)
        warmup = self.kmax - 1
        if warmup > 0:
            d2 = d2[:, warmup:, :]  # [B, L - warmup, K]

        return d2, warmup


# ============================================================
# 2) Surrogate LIF (NO refractory) with adaptation + hyperpolarization reset
# ============================================================
class LIF(nn.Module):
    def __init__(self, tau=20.0, threshold=1.0, tau_adapt=100.0, beta_adapt=1.5, v_reset=-0.5):
        super().__init__()
        self.tau, self.th, self.tau_adapt, self.beta_adapt, self.v_reset = tau, threshold, tau_adapt, beta_adapt, v_reset

    def forward(self, I):
        # I: [B, L]
        B, L = I.shape
        device = I.device
        dtype = I.dtype

        v = torch.zeros(B, 1, device=device, dtype=dtype)
        a = torch.zeros(B, 1, device=device, dtype=dtype)   # adaptation variable

        spikes = torch.zeros(B, L, device=device, dtype=dtype)
        spike_series = torch.zeros(B, L, device=device, dtype=dtype)
        s_num = torch.zeros(B, 1, device=device, dtype=dtype)
        k = 10.0  # surrogate slope

        for t in range(L):
            th_eff = self.th + self.beta_adapt * a

            dv = (-v / self.tau + I[:, t].unsqueeze(1))
            v = v + dv

            s_soft = torch.sigmoid(k * (v - th_eff))
            s_hard = (v >= th_eff).to(dtype)
            s = s_hard.detach() - s_soft.detach() + s_soft  # STE

            spikes[:, t] = s.squeeze(1)

            s_num = s_num + s
            spike_series[:, t] = s_num.squeeze(1)

            v = v * (1 - s.detach()) + self.v_reset * s.detach()

            da = -a / self.tau_adapt + s
            a = a + da

        return spikes, spike_series


# ============================================================
# 3) Winner-dominant competition (mechanics unchanged)
# ============================================================
class WinnerCombiner(nn.Module):
    """
    Winner-dominant competition with optional neuromorphic 'winner-trace' hysteresis.

    - d2[t,k] are dendritic / filter activations (>=0 from ReLU).
    - If trace is enabled: winner is computed from score = d2 + trace_gain * z
      where z is a leaky trace of past winners.
    - Winner remains HARD (argmax), we keep the same STE structure for weights.

    Returns:
      idx:  [B, L]     winner index per time
      vals: [B, L]     hard winner value (from d2, not score)
      w:    [B, L, K]  STE weights (hard forward)
      soma_in: [B, L]  weighted sum into soma (uses d2)
    """
    def __init__(self, temperature=0.1, inhibition_strength=1.0,
                 trace_tau=0.0, trace_gain=0.0):
        super().__init__()
        self.temperature = temperature
        self.inhibition_strength = inhibition_strength

        # trace_tau <= 0 disables trace
        self.trace_tau = float(trace_tau)
        self.trace_gain = float(trace_gain)

    def forward(self, d2):
        # d2: [B, L, K]
        B, L, K = d2.shape

        use_trace = (self.trace_tau > 0.0) and (self.trace_gain > 0.0)
        if use_trace:
            # leaky decay factor per step (stable and intuitive)
            # z <- decay*z + hard_winner
            decay = float(torch.exp(torch.tensor(-1.0 / self.trace_tau)).item())
            z = d2.new_zeros(B, K)  # [B, K] trace state

            idx_list = []
            w_sur_list = []

            for t in range(L):
                # score biases toward recent winners
                score_t = d2[:, t, :] + self.trace_gain * z  # [B, K]

                # surrogate soft weights from score (so surrogate matches the hard decision)
                logits = score_t / self.temperature
                soft_w = torch.softmax(logits, dim=-1)  # [B, K]

                mean_w = soft_w.mean(dim=-1, keepdim=True)
                inhibited = soft_w - self.inhibition_strength * mean_w
                inhibited = torch.relu(inhibited)
                w_sur = inhibited / (inhibited.sum(dim=-1, keepdim=True) + 1e-6)  # [B, K]
                w_sur_list.append(w_sur)

                # HARD winner from score (still hard!)
                idx_t = score_t.argmax(dim=-1)  # [B]
                idx_list.append(idx_t)

                # update trace with detached hard winner (no BPTT through discrete switches)
                hard_w_t = F.one_hot(idx_t, K).to(d2.dtype)  # [B, K]
                z = z * decay + hard_w_t.detach()

            idx = torch.stack(idx_list, dim=1)              # [B, L]
            w_sur = torch.stack(w_sur_list, dim=1)          # [B, L, K]

        else:
            # original behavior (no persistence)
            logits = d2 / self.temperature
            soft_w = torch.softmax(logits, dim=-1)  # [B, L, K]

            mean_w = soft_w.mean(dim=-1, keepdim=True)
            inhibited = soft_w - self.inhibition_strength * mean_w
            inhibited = torch.relu(inhibited)
            w_sur = inhibited / (inhibited.sum(dim=-1, keepdim=True) + 1e-6)

            idx = d2.argmax(dim=-1)  # [B, L]

        hard_w = F.one_hot(idx, K).to(d2.dtype)  # [B, L, K]

        # STE weights (hard forward, surrogate gradient path)
        w = hard_w.detach() - w_sur.detach() + w_sur  # [B, L, K]

        # soma input computed from actual dendritic activations (d2), not score
        soma_in = (w * d2).sum(dim=-1)      # [B, L]
        vals = (hard_w * d2).sum(dim=-1)    # [B, L] pure hard winner value from d2

        return idx, vals, w, soma_in


# ============================================================
# 4) Full model: ConvFilterBank -> WinnerCombiner -> LIF
# ============================================================
class LatencyPredictor(nn.Module):
    def __init__(self, kernel_sizes, temperature=0.1, inhibition_strength=1.0):
        super().__init__()
        self.fbA = ConvFilterBank(kernel_sizes=kernel_sizes)
        self.fbB = ConvFilterBank(kernel_sizes=kernel_sizes)

        self.encA = WinnerCombiner(temperature=temperature,
                            inhibition_strength=inhibition_strength,
                            trace_tau=8.0,        # z.B. 5..20
                            trace_gain=0.5)       # z.B. 0.1..1.0
        self.encB = WinnerCombiner(temperature=temperature,
                            inhibition_strength=inhibition_strength,
                            trace_tau=8.0,
                            trace_gain=0.5)

        # LIF WITHOUT refractory (kept as-is)
        self.lifA = LIF(tau=20.0, threshold=1.5, tau_adapt=100.0, beta_adapt=1.5, v_reset=-0.5)
        self.lifB = LIF(tau=20.0, threshold=1.5, tau_adapt=100.0, beta_adapt=1.5, v_reset=-0.5)

        self.kmax = max(kernel_sizes)

    def forward(self, xa, xb):
        # xa, xb: [B, L]
        d2a, warmup_a = self.fbA(xa)
        d2b, warmup_b = self.fbB(xb)
        warmup = max(warmup_a, warmup_b)

        ida, vala, wa, Ia = self.encA(d2a)
        idb, valb, wb, Ib = self.encB(d2b)

        sa, series_a = self.lifA(Ia)
        sb, series_b = self.lifB(Ib)

        return (d2a, ida, vala, wa, sa, series_a, warmup), \
               (d2b, idb, valb, wb, sb, series_b, warmup)


# ============================================================
# 5) Series-L1 loss (fully differentiable)
# ============================================================
def series_l1_loss(series_a, series_b):
    return torch.abs(series_a - series_b).mean()


# ============================================================
# 7) Generate synthetic self-supervised pair
# ============================================================
t = torch.linspace(0, 4 * 3.14, 400, device=device).unsqueeze(0)
clean = torch.sin(t)
noisy = clean + 0.2 * torch.randn_like(clean)


# ============================================================
# 8) Training loop
# ============================================================
kernel_sizes = [4, 6, 8, 10, 12, 20]  # variable filter lengths allowed (K = len(list))

model = LatencyPredictor(kernel_sizes=kernel_sizes, temperature=0.1, inhibition_strength=1.0).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
lambda_spike = 1e-3

for step in range(1000):
    (da, ida, vala, wa, sa, series_a, warmup), \
    (db, idb, valb, wb, sb, series_b, warmup2) = model(noisy, clean)

    loss = series_l1_loss(series_a, series_b) + lambda_spike * (sa.mean() + sb.mean())

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(float(loss))

    if step % 20 == 0:
        print(step, float(loss))


# ============================================================
# 6) Visualization utilities
# ============================================================
def plot_spike_rasters_with_winners(time, spikes_a, winner_idx_a, spikes_b, winner_idx_b, threshold=0.5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    configs = [(0, spikes_a, winner_idx_a, 'red', 'blue', 'Channel A (Noisy)'),
               (1, spikes_b, winner_idx_b, 'orange', 'green', 'Channel B (Clean)')]
    for ax_idx, spikes, winner_idx, spike_color, win_color, title in configs:
        ax = axes[ax_idx]
        ax.plot(time, spikes, color=spike_color, lw=0.2, alpha=0.7)
        ax.set_ylabel("Spike strength", color=spike_color)
        ax.tick_params(axis='y', labelcolor=spike_color)
        ax.set_title(title)
        if ax_idx == 1:
            ax.set_xlabel("Real time")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        spike_mask = spikes > threshold
        winner_on_spike = torch.where(spike_mask, winner_idx.float(), torch.nan)
        ax2.scatter(time[spike_mask], winner_on_spike[spike_mask], color=win_color, s=50, alpha=0.8)
        ax2.set_ylabel("Winning filter", color=win_color)
        ax2.tick_params(axis='y', labelcolor=win_color)
        ax2.set_ylim(-0.5, winner_idx.max().item() + 1)
    plt.tight_layout()
    plt.show()


def plot_dendrite_io(time, x, dendritic_outputs):
    # dendritic_outputs: [L, K]
    L, K = dendritic_outputs.shape
    fig, axes = plt.subplots(K, 2, figsize=(10, 2 * K))
    for k in range(K):
        axes[k, 0].plot(time, x, color='red')
        axes[k, 0].set_title(f"F{k} input")

        axes[k, 1].plot(time, dendritic_outputs[:, k], color='blue')
        axes[k, 1].set_title(f"F{k} output")

    plt.tight_layout()
    plt.show()


# ============================================================
# 6b) Spike-Centric "Causal Stack" plot
# ============================================================
def _rising_edges(spikes_1d, thr=0.5, refractory=3):
    s = spikes_1d.detach().cpu()
    above = (s > thr).int()
    rise = torch.where((above[1:] == 1) & (above[:-1] == 0))[0] + 1
    if rise.numel() == 0:
        return rise
    keep = [int(rise[0].item())]
    for idx in rise[1:].tolist():
        if idx - keep[-1] >= refractory:
            keep.append(idx)
    return torch.tensor(keep, dtype=torch.long)


def _colored_line(ax, x, y, c, lw=2.0, alpha=1.0):
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    c = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else c
    pts = list(zip(x, y))
    segs = [pts[i:i+2] for i in range(len(pts)-1)]
    lc = LineCollection(segs, array=c[:-1], linewidths=lw, alpha=alpha)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


def plot_causal_stack_per_spike(time, x, d2, w, spikes, winner_idx=None, tau=20.0, spike_thr=0.5,
                                window_factor=3.0, max_spikes=6, event_refractory=3, show_colorbar=True):
    """
    For each spike event (rising edge), plot:
      A) x(t) around spike
      B) contributions I_k(t)=w*d2 (ONLY winner + runner-up),
         colored along the curve by causal credit C_k(t)=I_k(t)*exp(-(t_s-t)/tau)
    """
    time_t = time.detach().cpu() if isinstance(time, torch.Tensor) else torch.tensor(time)
    x_t    = x.detach().cpu()    if isinstance(x, torch.Tensor) else torch.tensor(x)
    d2_t   = d2.detach().cpu()
    w_t    = w.detach().cpu()
    sp_t   = spikes.detach().cpu()
    win_t  = winner_idx.detach().cpu() if winner_idx is not None else None

    L, K = d2_t.shape
    tau_f = float(tau)
    win_len = int(window_factor * tau_f)

    I_d = w_t * d2_t  # [L, K]

    events = _rising_edges(sp_t, thr=spike_thr, refractory=event_refractory)
    if events.numel() == 0:
        print("No spike events (rising edges) found above threshold.")
        return
    events = events[:max_spikes]
    n = int(events.numel())

    all_credits = []
    for t_s in events.tolist():
        t0 = max(0, t_s - win_len)
        idx = torch.arange(t0, t_s + 1)
        decay = torch.exp(-(t_s - idx).float() / tau_f)
        C_all = I_d[idx, :] * decay.unsqueeze(1)
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

        decay = torch.exp(-(t_s - idx).float() / tau_f)
        C = I_d[idx, :] * decay.unsqueeze(1)

        k_star = int(win_t[t_s].item()) if win_t is not None else None

        totals = C.sum(dim=0)
        if k_star is not None and 0 <= k_star < K:
            totals_ru = totals.clone()
            totals_ru[k_star] = -1e9
            k_ru = int(torch.argmax(totals_ru).item())
        else:
            k_ru = int(torch.argmax(totals).item())
            k_star = k_ru

        axA = fig.add_subplot(gs[2*i + 0, 0])
        axA.plot(time_t[idx], x_t[idx], lw=1.2)
        axA.axvline(time_t[t_s].item(), lw=1.0)
        axA.set_ylabel("x(t)")
        axA.grid(True, alpha=0.25)
        axA.set_title(
            f"Spike {i+1} | t_idx={t_s} | winner F{k_star} | runner-up F{k_ru} | lookback≈{win_len}≈{window_factor}τ | τ={tau_f:g}"
        )

        axB = fig.add_subplot(gs[2*i + 1, 0], sharex=axA)

        lc1 = _colored_line(axB, time_t[idx], I_d[idx, k_star], C[:, k_star], lw=2.6, alpha=0.95)
        lc1.set_norm(norm)

        lc2 = _colored_line(axB, time_t[idx], I_d[idx, k_ru],   C[:, k_ru],   lw=1.6, alpha=0.85)
        lc2.set_norm(norm)

        axB.axvline(time_t[t_s].item(), lw=1.0)
        axB.set_ylabel("I_k(t)")
        axB.grid(True, alpha=0.25)

        if first_lc is None:
            first_lc = lc1

        if i == n - 1:
            axB.set_xlabel("time")

    if show_colorbar and first_lc is not None:
        fig.colorbar(first_lc, ax=fig.axes, fraction=0.02, pad=0.01, label="causal credit")

    plt.show()


# ============================================================
# 9) Visualization + example attribution
# ============================================================
if DO_PLOT:
    plt.figure(figsize=(10, 4))
    plt.plot(losses, lw=1)
    plt.xlabel("Training step")
    plt.ylabel("Loss (series L1 + spike cost)")
    plt.title("Training Progress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# time axis aligns to the valid region after warmup cropping
warmup = model.kmax - 1
L_eff = da.shape[1]
time = t[0, warmup:warmup + L_eff].detach().cpu()

if DO_PLOT:
    plot_spike_rasters_with_winners(
        time,
        sa[0].detach().cpu(), ida[0].detach().cpu(),
        sb[0].detach().cpu(), idb[0].detach().cpu()
    )

    plot_dendrite_io(
        time,
        noisy[0, warmup:warmup + L_eff].detach().cpu(),
        da[0].detach().cpu()
    )

    plot_causal_stack_per_spike(
        time=time,
        x=noisy[0, warmup:warmup + L_eff].detach().cpu(),
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
