import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0) DATA: long 2-sin + alternating segments (clean vs noise-only)
# ============================================================
def make_two_sin_alternating_long(
    T_total: int,
    t_max: float = 4 * math.pi,
    w1: float = 3.0,
    w2: float = 7.5,
    amp2: float = 0.6,
    phase2: float = 0.8,
    noise_std: float = 0.2,
    seg_len_range=(80, 200),
    start_with_noise: bool = False,
    device="cpu",
):
    """
    clean: [T_total]  = sin(w1*t) + amp2*sin(w2*t+phase2)
    noisy: [T_total]  = alternating segments:
                        - signal segment: noisy == clean
                        - noise segment: noisy is pure Gaussian noise (sin removed)
    mask_signal: [T_total] bool, True where noisy==clean
    """
    t = torch.linspace(0, t_max, T_total, device=device)
    clean = torch.sin(w1 * t) + amp2 * torch.sin(w2 * t + phase2)

    noisy = clean.clone()
    mask_signal = torch.ones(T_total, device=device, dtype=torch.bool)

    pos = 0
    is_noise = start_with_noise
    while pos < T_total:
        seg_len = int(torch.randint(seg_len_range[0], seg_len_range[1] + 1, (1,), device=device).item())
        end = min(T_total, pos + seg_len)

        if is_noise:
            noisy[pos:end] = noise_std * torch.randn(end - pos, device=device)
            mask_signal[pos:end] = False

        is_noise = not is_noise
        pos = end

    return clean, noisy, mask_signal


def build_sequence_bank(
    N_series: int,
    T_total: int,
    device="cpu",
    **gen_kwargs,
):
    clean_list, noisy_list, mask_list = [], [], []
    for _ in range(N_series):
        c, n, m = make_two_sin_alternating_long(T_total=T_total, device=device, **gen_kwargs)
        clean_list.append(c)
        noisy_list.append(n)
        mask_list.append(m)
    clean_bank = torch.stack(clean_list, dim=0)  # [N,T]
    noisy_bank = torch.stack(noisy_list, dim=0)  # [N,T]
    mask_bank = torch.stack(mask_list, dim=0)    # [N,T] bool
    return clean_bank, noisy_bank, mask_bank


def sample_random_windows_from_bank(clean_bank, noisy_bank, mask_bank, L: int, B: int, normalize_windows=True):
    """
    Returns:
      x: [B,L] noisy windows
      y: [B,L] clean windows
      m: [B,L] bool mask where x==y (signal)
    """
    device = clean_bank.device
    N_series, T_total = clean_bank.shape
    assert T_total >= L

    series_idx = torch.randint(0, N_series, (B,), device=device)
    starts = torch.randint(0, T_total - L + 1, (B,), device=device)

    y = torch.stack([clean_bank[series_idx[i], starts[i]:starts[i] + L] for i in range(B)], dim=0)
    x = torch.stack([noisy_bank[series_idx[i], starts[i]:starts[i] + L] for i in range(B)], dim=0)
    m = torch.stack([mask_bank[series_idx[i], starts[i]:starts[i] + L] for i in range(B)], dim=0)

    # NEW: per-window amplitude normalization (stabilizes dynamics)
    if normalize_windows:
        x = x / (x.abs().amax(dim=1, keepdim=True) + 1e-6)
        y = y / (y.abs().amax(dim=1, keepdim=True) + 1e-6)

    return x, y, m


# ============================================================
# 1) Hankel embedding + filterbank
# ============================================================
class HankelFilterBank(nn.Module):
    def __init__(self, hankel_size: int, K: int, bias: bool = True):
        super().__init__()
        self.H = int(hankel_size)
        self.K = int(K)
        self.lin = nn.Linear(self.H, self.K, bias=bias)

        # NEW: stable small init
        nn.init.normal_(self.lin.weight, mean=0.0, std=0.02)
        if self.lin.bias is not None:
            nn.init.constant_(self.lin.bias, 0.0)

    def forward(self, x):
        # x: [B,L]
        B, L = x.shape
        H = self.H

        xpad = F.pad(x.unsqueeze(1), (H - 1, 0))                  # [B,1,L+H-1]
        windows = xpad.unfold(dimension=2, size=H, step=1).squeeze(1)  # [B,L,H]

        # NEW: per-window normalization (very stabilizing)
        windows = (windows - windows.mean(dim=2, keepdim=True)) / (windows.std(dim=2, keepdim=True) + 1e-6)

        d2 = F.relu(self.lin(windows))                            # [B,L,K]

        warmup = H - 1
        if warmup > 0:
            d2 = d2[:, warmup:, :]
        return d2, warmup


# ============================================================
# 2) Per-filter Trace (with clamp)
# ============================================================
class PerFilterTrace(nn.Module):
    def __init__(self, lam=0.9, z_max=5.0):
        super().__init__()
        self.lam = float(lam)
        self.z_max = float(z_max)

    def forward(self, d2):
        B, L, K = d2.shape
        z = torch.zeros_like(d2)
        z_t = d2.new_zeros(B, K)
        lam = self.lam
        one_minus = 1.0 - lam
        for t in range(L):
            z_t = lam * z_t + one_minus * d2[:, t, :]
            # NEW: clamp to prevent runaway inf/nan
            z_t = torch.clamp(z_t, 0.0, self.z_max)
            z[:, t, :] = z_t
        return z


# ============================================================
# 3) MultiLIF
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
        device, dtype = I.device, I.dtype

        v = torch.zeros(B, K, device=device, dtype=dtype)
        a = torch.zeros(B, K, device=device, dtype=dtype)

        spikes = torch.zeros(B, L, K, device=device, dtype=dtype)
        series = torch.zeros(B, L, K, device=device, dtype=dtype)
        v_seq = torch.zeros(B, L, K, device=device, dtype=dtype)

        s_num = torch.zeros(B, K, device=device, dtype=dtype)
        k_surr = self.surr_slope

        for t in range(L):
            th_eff = self.th + self.beta_adapt * a
            v = v + (-v / self.tau + I[:, t, :])

            v_seq[:, t, :] = v  # snapshot before reset

            s_soft = torch.sigmoid(k_surr * (v - th_eff))
            s_hard = (v >= th_eff).to(dtype)
            s = s_hard.detach() - s_soft.detach() + s_soft  # STE

            spikes[:, t, :] = s
            s_num = s_num + s
            series[:, t, :] = s_num

            v = v * (1 - s.detach()) + self.v_reset * s.detach()
            a = a + (-a / self.tau_adapt + s)

        return spikes, series, v_seq


# ============================================================
# 4) WTA
# ============================================================
class MinimalFirstSpikeWTA(nn.Module):
    def __init__(self, temperature=0.2, thr=0.5):
        super().__init__()
        self.temperature = float(temperature)
        self.thr = float(thr)

    def forward(self, spikes):
        B, L, K = spikes.shape
        dtype = spikes.dtype
        device = spikes.device

        s = spikes > self.thr  # bool

        any_t = s.any(dim=2)
        has_any = any_t.any(dim=1)
        t_star = any_t.float().argmax(dim=1)

        s_at_t = s[torch.arange(B, device=device), t_star, :]
        k_star = s_at_t.float().argmax(dim=1)

        total = spikes.sum(dim=1)
        k_fallback = total.argmax(dim=1)
        idx = torch.where(has_any, k_star, k_fallback)

        w_hard = F.one_hot(idx, K).to(dtype)

        any_k = s.any(dim=1)
        c = s.int().cumsum(dim=1)
        first_mask = (c == 1) & s
        t_first = first_mask.float().argmax(dim=1)
        t_first = torch.where(any_k, t_first, torch.full_like(t_first, L))

        r = -t_first.to(dtype)
        w_sur = torch.softmax(r / self.temperature, dim=-1)

        w = w_hard.detach() - w_sur.detach() + w_sur
        spikes_gated = spikes * w.unsqueeze(1)
        return idx, w, spikes_gated


# ============================================================
# 5) Two-channel model
# ============================================================
class TwoChannelHankelTraceLIF_MinWTA(nn.Module):
    def __init__(
        self,
        hankel_size: int,
        K: int,
        trace_lam=0.9,
        trace_zmax=5.0,
        lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5,
        wta_temperature=0.2, wta_spike_thr=0.5,
    ):
        super().__init__()
        self.fbX = HankelFilterBank(hankel_size=hankel_size, K=K)
        self.fbY = HankelFilterBank(hankel_size=hankel_size, K=K)

        self.trX = PerFilterTrace(lam=trace_lam, z_max=trace_zmax)
        self.trY = PerFilterTrace(lam=trace_lam, z_max=trace_zmax)

        self.lifX = MultiLIF(tau=lif_tau, threshold=lif_th, tau_adapt=lif_tau_adapt,
                             beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)
        self.lifY = MultiLIF(tau=lif_tau, threshold=lif_th, tau_adapt=lif_tau_adapt,
                             beta_adapt=lif_beta_adapt, v_reset=lif_v_reset)

        self.wtaX = MinimalFirstSpikeWTA(temperature=wta_temperature, thr=wta_spike_thr)
        self.wtaY = MinimalFirstSpikeWTA(temperature=wta_temperature, thr=wta_spike_thr)

        self.warmup = hankel_size - 1
        self.K = K

    def forward(self, x, y):
        d2x, warmup_x = self.fbX(x)
        d2y, warmup_y = self.fbY(y)
        warmup = max(warmup_x, warmup_y)

        zx = self.trX(d2x)
        zy = self.trY(d2y)

        sx, series_x, vx = self.lifX(zx)
        sy, series_y, vy = self.lifY(zy)

        idx_x, w_x, sx_g = self.wtaX(sx)
        idx_y, w_y, sy_g = self.wtaY(sy)

        return (d2x, zx, sx, series_x, vx, idx_x, w_x, sx_g, warmup), \
               (d2y, zy, sy, series_y, vy, idx_y, w_y, sy_g, warmup)


# ============================================================
# 6) Loss with valid-mask + min_count
# ============================================================
def latency_jitter_loss_per_neuron(
    spikes: torch.Tensor,   # [B,L,K]
    v_seq: torch.Tensor,    # [B,L,K]
    idx: torch.Tensor,      # [B]
    lif_threshold: float,   # θ
    onset_frac: float = 0.6,
    alpha: float = 10.0,
    thr_spike: float = 0.5,
    lambda_mean: float = 0.0,
    min_count: int = 4,
):
    B, L, K = spikes.shape
    device = spikes.device
    dtype = v_seq.dtype
    b = torch.arange(B, device=device)

    s = spikes[b, :, idx]   # [B,L]
    v = v_seq[b, :, idx]    # [B,L]

    th_sp = float(lif_threshold)
    th_on = float(onset_frac) * float(lif_threshold)

    on_bool = (v >= th_on)
    sp_bool = (s > thr_spike)

    # NEW: only keep samples where onset and spike exist
    valid = on_bool.any(dim=1) & sp_bool.any(dim=1)

    # hard times
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
    delta_h = (t_sp_h - t_on_h).to(dtype)

    # soft times
    t = torch.arange(L, device=device, dtype=dtype).unsqueeze(0)

    a_on = torch.sigmoid(alpha * (v - th_on))
    p_on = a_on / (a_on.sum(dim=1, keepdim=True) + 1e-6)
    t_on_s = (p_on * t).sum(dim=1)

    a_sp = torch.sigmoid(alpha * (v - th_sp))
    p_sp = a_sp / (a_sp.sum(dim=1, keepdim=True) + 1e-6)
    t_sp_s = (p_sp * t).sum(dim=1)

    delta_s = t_sp_s - t_on_s
    delta = delta_h.detach() - delta_s.detach() + delta_s

    loss = delta.new_zeros(())
    mean_term = delta.new_zeros(())
    for k in range(K):
        m = (idx == k) & valid
        if int(m.sum().item()) >= min_count:
            dk = delta[m]
            loss = loss + dk.var(unbiased=False)
            mean_term = mean_term + dk.mean()

    return loss + lambda_mean * mean_term


# ============================================================
# 7) TRAINING LOOP (Phase 1) with all stabilizers
# ============================================================
def train_phase1(
    steps=2000,
    B=128,
    L=400,
    hankel_size=20,
    K=12,
    lr=1e-4,                 # NEW: more conservative
    device="cpu",
    onset_frac=0.6,
    alpha=10.0,
    lambda_mean=0.0,
    lambda_spike=1e-4,
    grad_clip=1.0,           # NEW
    # data bank
    N_series=64,
    T_total=20000,
    seed=0,
):
    torch.manual_seed(seed)

    # fixed data bank (generated once)
    clean_bank, noisy_bank, mask_bank = build_sequence_bank(
        N_series=N_series,
        T_total=T_total,
        device=device,
        t_max=4 * math.pi,
        w1=3.0, w2=7.5, amp2=0.6, phase2=0.8,
        noise_std=0.2,
        seg_len_range=(80, 200),
        start_with_noise=False,
    )

    model = TwoChannelHankelTraceLIF_MinWTA(
        hankel_size=hankel_size,
        K=K,
        trace_lam=0.9,
        trace_zmax=5.0,
        lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5,
        wta_temperature=0.2, wta_spike_thr=0.5,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        x, y, m = sample_random_windows_from_bank(
            clean_bank, noisy_bank, mask_bank,
            L=L, B=B,
            normalize_windows=True
        )

        (d2x, zx, sx, series_x, vx, idx_x, w_x, sx_g, warmup), \
        (d2y, zy, sy, series_y, vy, idx_y, w_y, sy_g, warmup2) = model(x, y)

        # PHASE 1: train ONLY channel X (recommended)
        loss_x = latency_jitter_loss_per_neuron(
            spikes=sx, v_seq=vx, idx=idx_x,
            lif_threshold=model.lifX.th,
            onset_frac=onset_frac, alpha=alpha,
            lambda_mean=lambda_mean,
            min_count=4,
        )

        spike_pen = (sx.mean() + sy.mean())
        loss = loss_x + lambda_spike * spike_pen

        # NaN guard with diagnostics
        if not torch.isfinite(loss):
            print("Non-finite loss at step", step)
            print("x finite:", torch.isfinite(x).all().item(),
                  "d2x finite:", torch.isfinite(d2x).all().item(),
                  "zx finite:", torch.isfinite(zx).all().item(),
                  "vx finite:", torch.isfinite(vx).all().item(),
                  "spikes finite:", torch.isfinite(sx).all().item())
            break

        opt.zero_grad()
        loss.backward()

        # NEW: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        opt.step()

        if step % 50 == 0:
            print(step, float(loss.detach().cpu()),
                  "| spike_mean:", float(spike_pen.detach().cpu()))

    return model


if __name__ == "__main__":
    train_phase1(
        steps=500,
        B=128,
        L=400,
        hankel_size=20,
        K=12,
        lr=1e-4,
        device="cpu",
        onset_frac=0.4,
        alpha=10.0,
        lambda_mean=0.0,
        lambda_spike=1e-4,
        grad_clip=1.0,
        N_series=64,
        T_total=20000,
        seed=0,
    )
