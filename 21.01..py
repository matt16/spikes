import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0) Synthetic data: ONE long 2-sine series + intermittent noise bursts
# ============================================================
def make_sine_burst_long(
    T: int,
    f1_range=(1.0, 3.0),
    f2_range=(5.0, 12.0),
    amp2=0.7,
    noise_std=0.15,
    burst_rate=0.02,          # expected burst-starts per timestep
    burst_std=0.8,
    burst_len_range=(10, 60),
    device="cpu",
):
    t = torch.linspace(0.0, 1.0, T, device=device)

    f1 = torch.empty((), device=device).uniform_(*f1_range)
    f2 = torch.empty((), device=device).uniform_(*f2_range)
    p1 = torch.empty((), device=device).uniform_(0, 2 * math.pi)
    p2 = torch.empty((), device=device).uniform_(0, 2 * math.pi)

    clean = torch.sin(2 * math.pi * f1 * t + p1) + amp2 * torch.sin(2 * math.pi * f2 * t + p2)
    noisy = clean + noise_std * torch.randn_like(clean)

    starts = (torch.rand(T, device=device) < burst_rate).nonzero(as_tuple=False).flatten()
    for s in starts.tolist():
        bl = int(torch.randint(burst_len_range[0], burst_len_range[1] + 1, (1,), device=device).item())
        e = min(T, s + bl)
        noisy[s:e] += burst_std * torch.randn(e - s, device=device)

    return clean, noisy  # [T], [T]


# ============================================================
# 0b) Window dataset over the long series
# ============================================================
class LongSeriesWindowDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_1d, clean_1d, window_len: int, stride: int = 1):
        assert noisy_1d.ndim == 1 and clean_1d.ndim == 1
        assert noisy_1d.shape == clean_1d.shape
        self.noisy = noisy_1d
        self.clean = clean_1d
        self.L = int(window_len)
        self.starts = torch.arange(0, noisy_1d.numel() - self.L + 1, stride, device="cpu")

    def __len__(self):
        return self.starts.numel()

    def __getitem__(self, i):
        s = int(self.starts[i].item())
        e = s + self.L
        return self.noisy[s:e], self.clean[s:e]  # [L], [L]


# ============================================================
# 1) Conv Filter Bank (causal) for x: [B,L] -> d2: [B,L_eff,K]
# ============================================================
class ConvFilterBank(nn.Module):
    def __init__(self, kernel_sizes, bias=True):
        super().__init__()
        ks = list(kernel_sizes)
        self.kmax = max(ks)

        size_to_count = {}
        for k in ks:
            size_to_count[k] = size_to_count.get(k, 0) + 1
        self._unique_sizes = sorted(size_to_count.keys())

        self.branches = nn.ModuleList([
            nn.Conv1d(1, size_to_count[k], kernel_size=k, stride=1, bias=bias)
            for k in self._unique_sizes
        ])
        self.K = sum(size_to_count[k] for k in self._unique_sizes)

    def forward(self, x):
        # x: [B,L]
        x1 = x.unsqueeze(1)  # [B,1,L]
        outs = []
        for conv, k in zip(self.branches, self._unique_sizes):
            outs.append(F.relu(conv(F.pad(x1, (k - 1, 0)))))  # [B,Ck,L]
        d2 = torch.cat(outs, 1).transpose(1, 2)  # [B,L,K]
        warmup = self.kmax - 1
        if warmup > 0:
            d2 = d2[:, warmup:, :]
        return d2, warmup


# ============================================================
# 2) Per-filter Trace
# ============================================================
class PerFilterTrace(nn.Module):
    def __init__(self, lam=0.95):
        super().__init__()
        self.lam = float(lam)

    def forward(self, d2):
        B, L, K = d2.shape
        z = torch.zeros_like(d2)
        z_t = d2.new_zeros(B, K)
        lam, one_minus = self.lam, 1.0 - self.lam
        for t in range(L):
            z_t = lam * z_t + one_minus * d2[:, t, :]
            z[:, t, :] = z_t
        return z


# ============================================================
# 3) MultiLIF (returns v_seq)
# ============================================================
class MultiLIF(nn.Module):
    def __init__(self, tau=20.0, threshold=1.5, tau_adapt=100.0, beta_adapt=1.5, v_reset=-0.5, surr_slope=10.0):
        super().__init__()
        self.tau = float(tau)
        self.th = float(threshold)
        self.tau_adapt = float(tau_adapt)
        self.beta_adapt = float(beta_adapt)
        self.v_reset = float(v_reset)
        self.surr_slope = float(surr_slope)

    def forward(self, I):
        B, L, K = I.shape
        v = torch.zeros(B, K, device=I.device, dtype=I.dtype)
        a = torch.zeros(B, K, device=I.device, dtype=I.dtype)
        spikes = torch.zeros(B, L, K, device=I.device, dtype=I.dtype)
        series = torch.zeros(B, L, K, device=I.device, dtype=I.dtype)
        v_seq = torch.zeros(B, L, K, device=I.device, dtype=I.dtype)
        s_num = torch.zeros(B, K, device=I.device, dtype=I.dtype)
        k_surr = self.surr_slope

        for t in range(L):
            th_eff = self.th + self.beta_adapt * a
            v = v + (-v / self.tau + I[:, t, :])
            v_seq[:, t, :] = v

            s_soft = torch.sigmoid(k_surr * (v - th_eff))
            s_hard = (v >= th_eff).to(I.dtype)
            s = s_hard.detach() - s_soft.detach() + s_soft

            spikes[:, t, :] = s
            s_num = s_num + s
            series[:, t, :] = s_num

            v = v * (1 - s.detach()) + self.v_reset * s.detach()
            a = a + (-a / self.tau_adapt + s)

        return spikes, series, v_seq


# ============================================================
# 4) Minimal First-Spike WTA
# ============================================================
class MinimalFirstSpikeWTA(nn.Module):
    def __init__(self, temperature=0.2, thr=0.5):
        super().__init__()
        self.temperature = float(temperature)
        self.thr = float(thr)

    def forward(self, spikes):
        B, L, K = spikes.shape
        s = spikes > self.thr

        any_t = s.any(2)
        has_any = any_t.any(1)
        t_star = any_t.float().argmax(1)

        s_at_t = s[torch.arange(B, device=spikes.device), t_star, :]
        k_star = s_at_t.float().argmax(1)

        total = spikes.sum(1)
        k_fallback = total.argmax(1)
        idx = torch.where(has_any, k_star, k_fallback)

        w_hard = F.one_hot(idx, K).to(spikes.dtype)

        any_k = s.any(1)
        c = s.int().cumsum(1)
        first_mask = (c == 1) & s
        t_first = first_mask.float().argmax(1)
        t_first = torch.where(any_k, t_first, torch.full_like(t_first, L))

        w_sur = torch.softmax((-t_first.to(spikes.dtype)) / self.temperature, dim=-1)
        w = w_hard.detach() - w_sur.detach() + w_sur
        return idx, w, spikes * w.unsqueeze(1)


# ============================================================
# 5) Two-channel model: ConvFB -> Trace -> LIF -> WTA
# ============================================================
class TwoChannel_Conv_TraceLIF_MinWTA(nn.Module):
    def __init__(
        self,
        kernel_sizes,
        trace_lam=0.95,
        lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5,
        wta_temperature=0.2, wta_spike_thr=0.5,
    ):
        super().__init__()
        self.fbX = ConvFilterBank(kernel_sizes=kernel_sizes)
        self.fbY = ConvFilterBank(kernel_sizes=kernel_sizes)

        self.trX = PerFilterTrace(lam=trace_lam)
        self.trY = PerFilterTrace(lam=trace_lam)

        self.lifX = MultiLIF(tau=lif_tau, threshold=lif_th, tau_adapt=lif_tau_adapt,
                             beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)
        self.lifY = MultiLIF(tau=lif_tau, threshold=lif_th, tau_adapt=lif_tau_adapt,
                             beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)

        self.wtaX = MinimalFirstSpikeWTA(temperature=wta_temperature, thr=wta_spike_thr)
        self.wtaY = MinimalFirstSpikeWTA(temperature=wta_temperature, thr=wta_spike_thr)

    def forward(self, x, y):
        d2x, wx = self.fbX(x)
        d2y, wy = self.fbY(y)
        warmup = max(wx, wy)

        zx = self.trX(d2x)
        zy = self.trY(d2y)

        sx, series_x, vx = self.lifX(zx)
        sy, series_y, vy = self.lifY(zy)

        idx_x, w_x, sx_g = self.wtaX(sx)
        idx_y, w_y, sy_g = self.wtaY(sy)

        return (d2x, zx, sx, series_x, vx, idx_x, w_x, sx_g, warmup), \
               (d2y, zy, sy, series_y, vy, idx_y, w_y, sy_g, warmup)


# ============================================================
# 6) Loss: winner latency jitter per neuron (no extra terms)
# ============================================================
def latency_jitter_loss_per_neuron(
    spikes: torch.Tensor,   # [B,L,K]
    v_seq: torch.Tensor,    # [B,L,K]
    idx: torch.Tensor,      # [B]
    lif_threshold: float,
    onset_frac: float = 0.6,
    alpha: float = 10.0,
    thr_spike: float = 0.5,
    min_count: int = 2,
):
    B, L, K = spikes.shape
    b = torch.arange(B, device=spikes.device)
    s = spikes[b, :, idx]
    v = v_seq[b, :, idx]

    th_sp = float(lif_threshold)
    th_on = float(onset_frac) * th_sp
    on, sp = (v >= th_on), (s > thr_spike)

    t_on_h = ((on.int().cumsum(1) == 1) & on).float().argmax(1)
    t_on_h = torch.where(on.any(1), t_on_h, torch.full_like(t_on_h, L))
    t_sp_h = ((sp.int().cumsum(1) == 1) & sp).float().argmax(1)
    t_sp_h = torch.where(sp.any(1), t_sp_h, torch.full_like(t_sp_h, L))

    delta_h = (t_sp_h.clamp(0, L - 1) - t_on_h.clamp(0, L - 1)).to(v.dtype)

    t = torch.arange(L, device=spikes.device, dtype=v.dtype).unsqueeze(0)
    a_on = torch.sigmoid(alpha * (v - th_on))
    a_sp = torch.sigmoid(alpha * (v - th_sp))
    t_on_s = (a_on / (a_on.sum(1, keepdim=True) + 1e-6) * t).sum(1)
    t_sp_s = (a_sp / (a_sp.sum(1, keepdim=True) + 1e-6) * t).sum(1)

    delta = delta_h.detach() - (t_sp_s - t_on_s).detach() + (t_sp_s - t_on_s)

    loss = delta.new_zeros(())
    for k in range(K):
        m = (idx == k)
        if int(m.sum()) >= min_count:
            loss = loss + delta[m].var(unbiased=False)
    return loss


# ============================================================
# 7) Training loop: long series -> windows -> DataLoader(shuffle)
# ============================================================
def train_phase1(
    steps=500,
    B=128,
    L=400,
    kernel_sizes=(3, 5, 9, 17),
    lr=4e-4,
    device="cpu",
    onset_frac=0.6,
    alpha=10.0,
    # long-series dataset params
    T_long=200_000,
    stride=5,
    shuffle=True,
    seed=0,
):
    torch.manual_seed(seed)

    model = TwoChannel_Conv_TraceLIF_MinWTA(
        kernel_sizes=kernel_sizes,
        trace_lam=0.95,
        lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5,
        wta_temperature=0.2, wta_spike_thr=0.5,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    clean_1d, noisy_1d = make_sine_burst_long(
        T=T_long,
        f1_range=(1.0, 3.0),
        f2_range=(6.0, 12.0),
        amp2=0.7,
        noise_std=0.12,
        burst_rate=0.02,
        burst_std=0.9,
        burst_len_range=(10, 60),
        device=device,
    )

    ds = LongSeriesWindowDataset(noisy_1d, clean_1d, window_len=L, stride=stride)
    dl = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=shuffle, drop_last=True)
    it = iter(dl)

    for step in range(steps):
        try:
            x, y = next(it)  # windows: [B,L]
        except StopIteration:
            it = iter(dl)
            x, y = next(it)

        (d2x, zx, sx, series_x, vx, idx_x, w_x, sx_g, warmup), \
        (d2y, zy, sy, series_y, vy, idx_y, w_y, sy_g, warmup2) = model(x, y)

        loss_x = latency_jitter_loss_per_neuron(sx, vx, idx_x, model.lifX.th, onset_frac, alpha)
        loss_y = latency_jitter_loss_per_neuron(sy, vy, idx_y, model.lifY.th, onset_frac, alpha)
        loss = loss_x + loss_y

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0:
            spike_mean = float((sx.mean() + sy.mean()).detach().cpu())
            print(step, float(loss.detach().cpu()), "| spike_mean:", spike_mean)

    return model


if __name__ == "__main__":
    train_phase1(
        steps=500,
        B=128,
        L=400,
        kernel_sizes=(3, 5, 9, 17),
        lr=4e-4,
        device="cpu",
        onset_frac=0.6,
        alpha=10.0,
        T_long=200_000,
        stride=5,
        shuffle=True,
        seed=0,
    )
