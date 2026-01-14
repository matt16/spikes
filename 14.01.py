import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) Causal Conv Filter Bank
# ============================================================
class ConvFilterBank(nn.Module):
    """
    Causal conv filter bank with variable kernel sizes.
    Output:
      d2: [B, L_eff, K] (ReLU), cropped by warmup = kmax-1
      warmup: int
    """
    def __init__(self, kernel_sizes, bias=True):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.kmax = max(self.kernel_sizes)

        size_to_count = {}
        for k in self.kernel_sizes:
            size_to_count[k] = size_to_count.get(k, 0) + 1

        self._unique_sizes = sorted(size_to_count.keys())
        self.branches = nn.ModuleList([
            nn.Conv1d(1, size_to_count[k], kernel_size=k, stride=1, bias=bias)
            for k in self._unique_sizes
        ])
        self.K = sum(size_to_count.values())

    def forward(self, x):
        # x: [B, L]
        x1 = x.unsqueeze(1)  # [B, 1, L]

        outs = []
        for conv, k in zip(self.branches, self._unique_sizes):
            xpad = F.pad(x1, (k - 1, 0))   # causal left padding
            y = conv(xpad)                 # [B, Ck, L]
            outs.append(F.relu(y))

        y_all = torch.cat(outs, dim=1)     # [B, K, L]
        d2 = y_all.transpose(1, 2)         # [B, L, K]

        warmup = self.kmax - 1
        if warmup > 0:
            d2 = d2[:, warmup:, :]         # [B, L_eff, K]

        return d2, warmup


# ============================================================
# 2) Per-filter Trace on d2
# ============================================================
class PerFilterTrace(nn.Module):
    """
    z_k(t) = lam*z_k(t-1) + (1-lam)*d2_k(t)
    """
    def __init__(self, lam=0.95):
        super().__init__()
        self.lam = float(lam)

    def forward(self, d2):
        B, L, K = d2.shape
        z = torch.zeros_like(d2)
        z_t = d2.new_zeros(B, K)

        lam = self.lam
        one_minus = 1.0 - lam
        for t in range(L):
            z_t = lam * z_t + one_minus * d2[:, t, :]
            z[:, t, :] = z_t
        return z


# ============================================================
# 3) MultiLIF (minimal FIX): returns v_seq for onset measurement
# ============================================================
class MultiLIF(nn.Module):
    """
    Input I: [B, L, K]
    Returns:
      spikes:       [B, L, K]  (hard with STE)
      spike_series: [B, L, K]
      v_seq:        [B, L, K]  membrane BEFORE reset (for latency onset)
    """
    def __init__(self, tau=20.0, threshold=1.0, tau_adapt=100.0, beta_adapt=1.5, v_reset=-0.5, surr_slope=10.0):
        super().__init__()
        self.tau = float(tau)
        self.th = float(threshold)
        self.tau_adapt = float(tau_adapt)
        self.beta_adapt = float(beta_adapt)
        self.v_reset = float(v_reset)
        self.surr_slope = float(surr_slope)

    def forward(self, I):
        B, L, K = I.shape
        device, dtype = I.device, I.dtype

        v = torch.zeros(B, K, device=device, dtype=dtype)
        a = torch.zeros(B, K, device=device, dtype=dtype)

        spikes = torch.zeros(B, L, K, device=device, dtype=dtype)
        spike_series = torch.zeros(B, L, K, device=device, dtype=dtype)
        v_seq = torch.zeros(B, L, K, device=device, dtype=dtype)

        s_num = torch.zeros(B, K, device=device, dtype=dtype)
        k_surr = self.surr_slope

        for t in range(L):
            th_eff = self.th + self.beta_adapt * a

            v = v + (-v / self.tau + I[:, t, :])

            # snapshot BEFORE reset (needed for onset)
            v_seq[:, t, :] = v

            s_soft = torch.sigmoid(k_surr * (v - th_eff))
            s_hard = (v >= th_eff).to(dtype)
            s = s_hard.detach() - s_soft.detach() + s_soft  # STE

            spikes[:, t, :] = s
            s_num = s_num + s
            spike_series[:, t, :] = s_num

            # reset
            v = v * (1 - s.detach()) + self.v_reset * s.detach()

            da = -a / self.tau_adapt + s
            a = a + da

        return spikes, spike_series, v_seq


# ============================================================
# 4) Minimal WTA after LIF: global first spike wins + relative soft surrogate
# ============================================================
class MinimalFirstSpikeWTA(nn.Module):
    """
    Global first-spike WTA per SAMPLE:
      - Forward: pick (t*, k*) = earliest spike over ALL filters; silence others.
      - Backward: w_sur = softmax(-t_first / T) (relative timing only).
    """
    def __init__(self, temperature=0.1, thr=0.5):
        super().__init__()
        self.temperature = float(temperature)
        self.thr = float(thr)

    def forward(self, spikes):
        # spikes: [B, L, K]
        B, L, K = spikes.shape
        dtype = spikes.dtype
        device = spikes.device

        s = spikes > self.thr  # bool [B,L,K]

        # hard: global earliest time with any spike
        any_t = s.any(dim=2)          # [B,L]
        has_any = any_t.any(dim=1)    # [B]
        t_star = any_t.float().argmax(dim=1)  # [B] (0 if none)

        # hard: pick first filter spiking at t_star
        s_at_t = s[torch.arange(B, device=device), t_star, :]  # [B,K]
        k_star = s_at_t.float().argmax(dim=1)                  # [B] (0 if none)

        # fallback if no spikes
        total = spikes.sum(dim=1)  # [B,K]
        k_fallback = total.argmax(dim=1)
        idx = torch.where(has_any, k_star, k_fallback)         # [B]

        w_hard = F.one_hot(idx, K).to(dtype)  # [B,K]

        # soft: compute first spike time per filter (L if none) -> purely relative timing
        any_k = s.any(dim=1)                  # [B,K]
        c = s.int().cumsum(dim=1)
        first_mask = (c == 1) & s
        t_first = first_mask.float().argmax(dim=1)             # [B,K] (0 if none)
        t_first = torch.where(any_k, t_first, torch.full_like(t_first, L))  # [B,K]

        r = -t_first.to(dtype)                                 # earlier => larger
        w_sur = torch.softmax(r / self.temperature, dim=-1)    # [B,K]

        w = w_hard.detach() - w_sur.detach() + w_sur           # STE
        spikes_gated = spikes * w.unsqueeze(1)                 # [B,L,K]
        return idx, w, spikes_gated


# ============================================================
# 5) Full model: ConvFilterBank -> Trace -> MultiLIF -> WTA (2 channels)
# ============================================================
class TwoChannelTraceLIF_MinWTA(nn.Module):
    """
    Returns per channel:
      (d2, z, spikes_raw, series_raw, v_seq, winner_idx, w, spikes_gated, warmup)
    """
    def __init__(
        self,
        kernel_sizes,
        trace_lam=0.95,
        lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5,
        wta_temperature=0.2, wta_spike_thr=0.5
    ):
        super().__init__()
        self.fbX = ConvFilterBank(kernel_sizes=kernel_sizes)
        self.fbY = ConvFilterBank(kernel_sizes=kernel_sizes)

        self.trX = PerFilterTrace(lam=trace_lam)
        self.trY = PerFilterTrace(lam=trace_lam)

        self.lifX = MultiLIF(tau=lif_tau, threshold=lif_th,
                             tau_adapt=lif_tau_adapt, beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)
        self.lifY = MultiLIF(tau=lif_tau, threshold=lif_th,
                             tau_adapt=lif_tau_adapt, beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)

        self.wtaX = MinimalFirstSpikeWTA(temperature=wta_temperature, thr=wta_spike_thr)
        self.wtaY = MinimalFirstSpikeWTA(temperature=wta_temperature, thr=wta_spike_thr)

        self.kmax = max(kernel_sizes)

    def forward(self, x, y):
        # x, y: [B,L]
        d2x, warmup_x = self.fbX(x)  # [B,L_eff,K]
        d2y, warmup_y = self.fbY(y)
        warmup = max(warmup_x, warmup_y)

        zx = self.trX(d2x)
        zy = self.trY(d2y)

        sx, series_x, vx = self.lifX(zx)
        sy, series_y, vy = self.lifY(zy)

        idx_x, w_x, sx_gated = self.wtaX(sx)
        idx_y, w_y, sy_gated = self.wtaY(sy)

        return (
            d2x, zx, sx, series_x, vx, idx_x, w_x, sx_gated, warmup
        ), (
            d2y, zy, sy, series_y, vy, idx_y, w_y, sy_gated, warmup
        )


# ============================================================
# 6) Phase-1 loss: per-neuron variance of Δ = t_spike - t_onset (winner-only)
#     hard forward, soft backward (STE on Δ)
# ============================================================
def latency_jitter_loss_per_neuron(
    spikes: torch.Tensor,   # [B,L,K]
    v_seq: torch.Tensor,    # [B,L,K]
    idx: torch.Tensor,      # [B]
    lif_threshold: float,   # θ (use model.lifX.th or lifY.th)
    onset_frac: float = 0.6,
    alpha: float = 10.0,
    thr_spike: float = 0.5,
    lambda_mean: float = 0.0,
    min_count: int = 2,
):
    B, L, K = spikes.shape
    device = spikes.device
    dtype = v_seq.dtype

    b = torch.arange(B, device=device)

    # winner traces: [B,L]
    s = spikes[b, :, idx]
    v = v_seq[b, :, idx]

    th_sp = float(lif_threshold)
    th_on = float(onset_frac) * float(lif_threshold)

    # ---------- HARD times ----------
    on_bool = (v >= th_on)            # [B,L]
    sp_bool = (s > thr_spike)         # [B,L]

    c_on = on_bool.int().cumsum(dim=1)
    first_on = (c_on == 1) & on_bool
    t_on_h = first_on.float().argmax(dim=1)
    t_on_h = torch.where(on_bool.any(dim=1), t_on_h, torch.full_like(t_on_h, L))

    c_sp = sp_bool.int().cumsum(dim=1)
    first_sp = (c_sp == 1) & sp_bool
    t_sp_h = first_sp.float().argmax(dim=1)
    t_sp_h = torch.where(sp_bool.any(dim=1), t_sp_h, torch.full_like(t_sp_h, L))

    t_on_h = torch.clamp(t_on_h, 0, L - 1)
    t_sp_h = torch.clamp(t_sp_h, 0, L - 1)
    delta_h = (t_sp_h - t_on_h).to(dtype)  # [B]

    # ---------- SOFT times ----------
    t = torch.arange(L, device=device, dtype=dtype).unsqueeze(0)  # [1,L]

    a_on = torch.sigmoid(alpha * (v - th_on))                     # [B,L]
    p_on = a_on / (a_on.sum(dim=1, keepdim=True) + 1e-6)
    t_on_s = (p_on * t).sum(dim=1)                                # [B]

    a_sp = torch.sigmoid(alpha * (v - th_sp))                     # [B,L]
    p_sp = a_sp / (a_sp.sum(dim=1, keepdim=True) + 1e-6)
    t_sp_s = (p_sp * t).sum(dim=1)                                # [B]

    delta_s = t_sp_s - t_on_s

    # STE on delta: forward uses hard Δ, backward uses soft Δ
    delta = delta_h.detach() - delta_s.detach() + delta_s         # [B]

    # ---------- per-neuron variance over samples where idx==k ----------
    loss = delta.new_zeros(())
    mean_term = delta.new_zeros(())
    for k in range(K):
        m = (idx == k)
        n = int(m.sum().item())
        if n >= min_count:
            dk = delta[m]
            loss = loss + dk.var(unbiased=False)
            mean_term = mean_term + dk.mean()

    return loss + lambda_mean * mean_term


# ============================================================
# 7) Minimal training step wrapper (no class)
# ============================================================
def train_step_phase1(
    model: TwoChannelTraceLIF_MinWTA,
    opt: torch.optim.Optimizer,
    x: torch.Tensor, y: torch.Tensor,
    onset_frac: float = 0.6,
    alpha: float = 10.0,
    lambda_mean: float = 0.0,
    lambda_spike: float = 1e-4,
    thr_spike: float = 0.5,
):
    (d2x, zx, sx, series_x, vx, idx_x, w_x, sx_gated, warmup), \
    (d2y, zy, sy, series_y, vy, idx_y, w_y, sy_gated, warmup2) = model(x, y)

    loss_x = latency_jitter_loss_per_neuron(
        spikes=sx, v_seq=vx, idx=idx_x,
        lif_threshold=model.lifX.th,
        onset_frac=onset_frac, alpha=alpha,
        thr_spike=thr_spike, lambda_mean=lambda_mean
    )
    loss_y = latency_jitter_loss_per_neuron(
        spikes=sy, v_seq=vy, idx=idx_y,
        lif_threshold=model.lifY.th,
        onset_frac=onset_frac, alpha=alpha,
        thr_spike=thr_spike, lambda_mean=lambda_mean
    )

    spike_pen = (sx.mean() + sy.mean())
    loss = loss_x + loss_y + lambda_spike * spike_pen

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss


# ============================================================
# 8) Tiny example run
# ============================================================
if __name__ == "__main__":
    device = "cpu"

    kernel_sizes = [4, 6, 8, 10, 12, 20]
    model = TwoChannelTraceLIF_MinWTA(
        kernel_sizes=kernel_sizes,
        trace_lam=0.95,
        lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5,
        wta_temperature=0.2, wta_spike_thr=0.5
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    B, L = 8, 400
    x = torch.randn(B, L, device=device)
    y = torch.randn(B, L, device=device)

    loss = train_step_phase1(model, opt, x, y, onset_frac=0.6, alpha=10.0, lambda_mean=0.0, lambda_spike=1e-4)
    print("loss:", float(loss.detach().cpu()))
