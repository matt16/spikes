import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) Causal Conv Filter Bank (as in your code)
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
        B, L = x.shape
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
            d2 = d2[:, warmup:, :]         # [B, L-warmup, K]

        return d2, warmup


# ============================================================
# 2) Per-filter Trace on d2: z_k(t)=λ z_k(t-1) + (1-λ) d2_k(t)
# ============================================================
class PerFilterTrace(nn.Module):
    """
    Applies a leaky trace over time independently per filter.
    Input:  d2 [B, L, K]
    Output: z  [B, L, K]
    """
    def __init__(self, lam=0.95):
        super().__init__()
        self.lam = float(lam)

    def forward(self, d2):
        B, L, K = d2.shape
        z = torch.zeros_like(d2)
        z_t = d2.new_zeros(B, K)  # state per filter

        lam = self.lam
        one_minus = 1.0 - lam

        for t in range(L):
            z_t = lam * z_t + one_minus * d2[:, t, :]
            z[:, t, :] = z_t

        return z


# ============================================================
# 3) MultiLIF: your LIF adapted to [B, L, K]
# ============================================================
class MultiLIF(nn.Module):
    """
    Vectorized version of your LIF:
    - Separate membrane/adaptation state per filter k
    - Input I: [B, L, K]
    - Output spikes:      [B, L, K]
            spike_series:[B, L, K]  (cumulative count per filter)
    """
    def __init__(self, tau=20.0, threshold=1.0, tau_adapt=100.0, beta_adapt=1.5, v_reset=-0.5):
        super().__init__()
        self.tau = float(tau)
        self.th = float(threshold)
        self.tau_adapt = float(tau_adapt)
        self.beta_adapt = float(beta_adapt)
        self.v_reset = float(v_reset)

    def forward(self, I):
        # I: [B, L, K]
        B, L, K = I.shape
        device = I.device
        dtype = I.dtype

        v = torch.zeros(B, K, device=device, dtype=dtype)
        a = torch.zeros(B, K, device=device, dtype=dtype)

        spikes = torch.zeros(B, L, K, device=device, dtype=dtype)
        spike_series = torch.zeros(B, L, K, device=device, dtype=dtype)
        s_num = torch.zeros(B, K, device=device, dtype=dtype)

        k_surr = 10.0  # surrogate slope

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
# 4) Full model: ConvFilterBank -> Trace -> MultiLIF (2 channels: x and y)
# ============================================================
class TwoChannelTraceLIF(nn.Module):
    """
    New architecture (no training loop yet):
      x -> ConvFilterBank -> d2x -> Trace -> zx -> MultiLIF -> spikes_x, series_x
      y -> ConvFilterBank -> d2y -> Trace -> zy -> MultiLIF -> spikes_y, series_y

    Returns tuples per channel: (d2, z, spikes, series, warmup)
    """
    def __init__(self, kernel_sizes, trace_lam=0.95,
                 lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5):
        super().__init__()
        self.fbX = ConvFilterBank(kernel_sizes=kernel_sizes)
        self.fbY = ConvFilterBank(kernel_sizes=kernel_sizes)

        self.trX = PerFilterTrace(lam=trace_lam)
        self.trY = PerFilterTrace(lam=trace_lam)

        self.lifX = MultiLIF(tau=lif_tau, threshold=lif_th,
                             tau_adapt=lif_tau_adapt, beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)
        self.lifY = MultiLIF(tau=lif_tau, threshold=lif_th,
                             tau_adapt=lif_tau_adapt, beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)

        self.kmax = max(kernel_sizes)

    def forward(self, x, y):
        # x, y: [B, L]
        d2x, warmup_x = self.fbX(x)   # [B, L_eff, K]
        d2y, warmup_y = self.fbY(y)

        warmup = max(warmup_x, warmup_y)

        zx = self.trX(d2x)            # [B, L_eff, K]
        zy = self.trY(d2y)

        sx, series_x = self.lifX(zx)  # [B, L_eff, K]
        sy, series_y = self.lifY(zy)

        return (d2x, zx, sx, series_x, warmup), (d2y, zy, sy, series_y, warmup)
