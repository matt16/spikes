import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) Causal Conv Filter Bank (unchanged)
# ============================================================
class ConvFilterBank(nn.Module):
    """
    Runs a bank of causal 1D conv filters with stride=1.
    Output:
      d2: [B, L_eff, K]  (ReLU activations, cropped to valid region based on max kernel)
      warmup: int = (max_kernel_size - 1)
    """
    def __init__(self, kernel_sizes, bias=True):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.kmax = max(self.kernel_sizes)

        size_to_count = {}
        for k in self.kernel_sizes:
            size_to_count[k] = size_to_count.get(k, 0) + 1

        self._unique_sizes = sorted(size_to_count.keys())
        self._counts = [size_to_count[k] for k in self._unique_sizes]

        self.branches = nn.ModuleList([
            nn.Conv1d(1, size_to_count[k], kernel_size=k, stride=1, bias=bias)
            for k in self._unique_sizes
        ])

        self.K = sum(self._counts)

    def forward(self, x):
        # x: [B, L]
        x1 = x.unsqueeze(1)  # [B, 1, L]

        outs = []
        for conv, k in zip(self.branches, self._unique_sizes):
            xpad = F.pad(x1, (k - 1, 0))   # causal left padding
            y = conv(xpad)                 # [B, Ck, L]
            y = F.relu(y)
            outs.append(y)

        y_all = torch.cat(outs, dim=1)     # [B, K, L]
        d2 = y_all.transpose(1, 2)         # [B, L, K]

        warmup = self.kmax - 1
        if warmup > 0:
            d2 = d2[:, warmup:, :]         # [B, L_eff, K]

        return d2, warmup


# ============================================================
# 2) Per-filter Trace on d2 (unchanged)
# ============================================================
class PerFilterTrace(nn.Module):
    """
    z_k(t) = lam*z_k(t-1) + (1-lam)*d2_k(t)
    Input:  d2 [B, L, K]
    Output: z  [B, L, K]
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
# 3) MultiLIF: your LIF adapted to [B, L, K] (minimal, no extras)
# ============================================================
class MultiLIF(nn.Module):
    """
    Vectorized version of your LIF:
      Input I: [B, L, K]
      Returns:
        spikes:       [B, L, K]  (hard with STE)
        spike_series: [B, L, K]  (cumulative per filter)
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
        # I: [B, L, K]
        B, L, K = I.shape
        device, dtype = I.device, I.dtype

        v = torch.zeros(B, K, device=device, dtype=dtype)
        a = torch.zeros(B, K, device=device, dtype=dtype)

        spikes = torch.zeros(B, L, K, device=device, dtype=dtype)
        spike_series = torch.zeros(B, L, K, device=device, dtype=dtype)
        s_num = torch.zeros(B, K, device=device, dtype=dtype)

        k_surr = self.surr_slope

        for t in range(L):
            th_eff = self.th + self.beta_adapt * a

            dv = (-v / self.tau + I[:, t, :])
            v = v + dv

            s_soft = torch.sigmoid(k_surr * (v - th_eff))
            s_hard = (v >= th_eff).to(dtype)
            s = s_hard.detach() - s_soft.detach() + s_soft  # STE

            spikes[:, t, :] = s

            s_num = s_num + s
            spike_series[:, t, :] = s_num

            v = v * (1 - s.detach()) + self.v_reset * s.detach()

            da = -a / self.tau_adapt + s
            a = a + da

        return spikes, spike_series


# ============================================================
# 4) Minimal WTA after LIF: earliest spike wins + soft surrogate
# ============================================================
class MinimalFirstSpikeWTA(nn.Module):
    """
    Minimal WTA after LIF, per SAMPLE:
      - Hard forward: filter with earliest hard spike wins; all others silenced.
      - Soft backward: early-evidence score from spikes, softmax + STE.

    Input:
      spikes: [B, L, K]
    Output:
      idx:          [B]
      w:            [B, K] (STE)
      spikes_gated: [B, L, K]
    """
    def __init__(self, tau_early=20.0, temperature=0.2, thr=0.5, eps=1e-6):
        super().__init__()
        self.tau_early = float(tau_early)
        self.temperature = float(temperature)
        self.thr = float(thr)
        self.eps = float(eps)

    @staticmethod
    def _first_spike_time(spikes_bool):
        # spikes_bool: [B, L, K] bool
        # returns t_first: [B, K] where L means "no spike"
        B, L, K = spikes_bool.shape
        any_spk = spikes_bool.any(dim=1)  # [B, K]

        c = spikes_bool.int().cumsum(dim=1)         # [B, L, K]
        first_mask = (c == 1) & spikes_bool         # [B, L, K]
        t_first = first_mask.float().argmax(dim=1)  # [B, K] (0 if none)
        t_first = torch.where(any_spk, t_first, torch.full_like(t_first, L))
        return t_first

    def forward(self, spikes):
        B, L, K = spikes.shape
        device, dtype = spikes.device, spikes.dtype

        # HARD winner: earliest spike
        spikes_bool = spikes > self.thr
        t_first = self._first_spike_time(spikes_bool)   # [B, K], L if none
        idx_hard = torch.argmin(t_first, dim=-1)        # [B]

        # If nobody spikes in a sample: fallback to largest total spikes (often all 0 -> ties -> 0)
        no_spike = (t_first.min(dim=-1).values >= L)    # [B]
        total = spikes.sum(dim=1)                       # [B, K]
        idx_fallback = torch.argmax(total, dim=-1)      # [B]
        idx = torch.where(no_spike, idx_fallback, idx_hard)

        w_hard = F.one_hot(idx, K).to(dtype)            # [B, K]

        # SOFT winner: early-evidence score
        t = torch.arange(L, device=device, dtype=dtype).view(1, L, 1)
        decay = torch.exp(-t / max(self.tau_early, self.eps))          # [1, L, 1]
        r = (spikes * decay).sum(dim=1)                                # [B, K]

        w_sur = torch.softmax(r / self.temperature, dim=-1)            # [B, K]

        # STE weights
        w = w_hard.detach() - w_sur.detach() + w_sur                   # [B, K]

        # Gate spikes (broadcast over time)
        spikes_gated = spikes * w.unsqueeze(1)                         # [B, L, K]

        return idx, w, spikes_gated


# ============================================================
# 5) Full model: ConvFilterBank -> Trace -> MultiLIF -> MinimalFirstSpikeWTA
#    (Two channels: x and y, different filterbanks)
# ============================================================
class TwoChannelTraceLIF_MinWTA(nn.Module):
    """
    Per channel:
      x/y -> ConvFilterBank -> d2 -> Trace(z) -> MultiLIF -> spikes -> WTA -> spikes_gated

    Returns per channel:
      (d2, z, spikes_raw, series_raw, winner_idx, w, spikes_gated, warmup)
    """
    def __init__(self,
                 kernel_sizes,
                 trace_lam=0.95,
                 lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5,
                 wta_tau_early=20.0, wta_temperature=0.2, wta_spike_thr=0.5):
        super().__init__()
        # different filterbanks as requested
        self.fbX = ConvFilterBank(kernel_sizes=kernel_sizes)
        self.fbY = ConvFilterBank(kernel_sizes=kernel_sizes)

        self.trX = PerFilterTrace(lam=trace_lam)
        self.trY = PerFilterTrace(lam=trace_lam)

        self.lifX = MultiLIF(tau=lif_tau, threshold=lif_th,
                             tau_adapt=lif_tau_adapt, beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)
        self.lifY = MultiLIF(tau=lif_tau, threshold=lif_th,
                             tau_adapt=lif_tau_adapt, beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)

        self.wtaX = MinimalFirstSpikeWTA(tau_early=wta_tau_early, temperature=wta_temperature, thr=wta_spike_thr)
        self.wtaY = MinimalFirstSpikeWTA(tau_early=wta_tau_early, temperature=wta_temperature, thr=wta_spike_thr)

        self.kmax = max(kernel_sizes)

    def forward(self, x, y):
        # x, y: [B, L]
        d2x, warmup_x = self.fbX(x)  # [B, L_eff, K]
        d2y, warmup_y = self.fbY(y)

        warmup = max(warmup_x, warmup_y)

        zx = self.trX(d2x)           # [B, L_eff, K]
        zy = self.trY(d2y)

        sx, series_x = self.lifX(zx) # [B, L_eff, K]
        sy, series_y = self.lifY(zy)

        idx_x, w_x, sx_gated = self.wtaX(sx)
        idx_y, w_y, sy_gated = self.wtaY(sy)

        return (
            d2x, zx, sx, series_x, idx_x, w_x, sx_gated, warmup
        ), (
            d2y, zy, sy, series_y, idx_y, w_y, sy_gated, warmup
        )
