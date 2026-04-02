import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0) Synthetic data: 2-sine mixture + intermittent noise bursts
# ============================================================
def make_sine_burst_batch(
    B: int,
    L: int,
    f1_range=(1.0, 3.0),
    f2_range=(5.0, 12.0),
    amp2=0.7,
    noise_std=0.15,
    burst_prob=0.8,
    burst_std=0.8,
    burst_len_range=(10, 60),
    device="cpu",
):
    """
    Returns:
      clean: [B,L]
      noisy: [B,L]  (clean + base noise + burst noise segments)
    """
    t = torch.linspace(0.0, 1.0, L, device=device).unsqueeze(0).repeat(B, 1)  # [B,L]

    f1 = torch.empty(B, 1, device=device).uniform_(*f1_range)
    f2 = torch.empty(B, 1, device=device).uniform_(*f2_range)

    # random phase
    p1 = torch.empty(B, 1, device=device).uniform_(0, 2 * math.pi)
    p2 = torch.empty(B, 1, device=device).uniform_(0, 2 * math.pi)

    clean = torch.sin(2 * math.pi * f1 * t + p1) + amp2 * torch.sin(2 * math.pi * f2 * t + p2)

    # base noise
    noisy = clean + noise_std * torch.randn_like(clean)

    # noise bursts
    if burst_prob > 0:
        for b in range(B):
            if torch.rand((), device=device).item() < burst_prob:
                burst_len = int(torch.randint(burst_len_range[0], burst_len_range[1] + 1, (1,), device=device).item())
                start = int(torch.randint(0, max(1, L - burst_len), (1,), device=device).item())
                noisy[b, start:start + burst_len] += burst_std * torch.randn(burst_len, device=device)

    return clean, noisy


# ============================================================
# 1) Hankel embedding (causal) + learned "filterbank" over Hankel vectors
# ============================================================
class HankelFilterBank(nn.Module):
    """
    Creates causal Hankel vectors of length H for each time t:
      h(t) = [x[t-H+1], ..., x[t]]   (zero padded on left)
    Then applies a learned linear map to K filters:
      d2(t,k) = ReLU( sum_i W[k,i] * h(t,i) + b[k] )

    Output:
      d2: [B, L_eff, K]
      warmup: int = H-1  (we crop first H-1 steps for fairness)
    """
    def __init__(self, hankel_size: int, K: int, bias: bool = True):
        super().__init__()
        self.H = int(hankel_size)
        self.K = int(K)
        self.lin = nn.Linear(self.H, self.K, bias=bias)

    def forward(self, x):
        # x: [B,L]
        B, L = x.shape
        H = self.H

        # causal left padding, then unfold into Hankel windows
        xpad = F.pad(x.unsqueeze(1), (H - 1, 0))          # [B,1,L+H-1]
        windows = xpad.unfold(dimension=2, size=H, step=1)  # [B,1,L,H]
        windows = windows.squeeze(1)                        # [B,L,H]

        d2 = F.relu(self.lin(windows))  # [B,L,K]

        warmup = H - 1
        if warmup > 0:
            d2 = d2[:, warmup:, :]      # [B, L_eff, K]

        return d2, warmup


# ============================================================
# 2) Per-filter Trace on d2
# ============================================================
class PerFilterTrace(nn.Module):
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
# 3) MultiLIF (minimal fix): returns v_seq
# ============================================================
class MultiLIF(nn.Module):
    """
    Input I: [B,L,K]
    Returns:
      spikes: [B,L,K] hard with STE
      series: [B,L,K]
      v_seq:  [B,L,K] membrane BEFORE reset (for onset measurement)
    """
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

            v_seq[:, t, :] = v  # snapshot BEFORE reset

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
# 4) Minimal WTA after LIF: global first spike wins + relative soft surrogate
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

        s = spikes > self.thr  # bool [B,L,K]

        # hard global first-spike winner
        any_t = s.any(dim=2)          # [B,L]
        has_any = any_t.any(dim=1)    # [B]
        t_star = any_t.float().argmax(dim=1)  # [B] (0 if none)

        s_at_t = s[torch.arange(B, device=device), t_star, :]  # [B,K]
        k_star = s_at_t.float().argmax(dim=1)                  # [B]

        # fallback if no spikes
        total = spikes.sum(dim=1)  # [B,K]
        k_fallback = total.argmax(dim=1)
        idx = torch.where(has_any, k_star, k_fallback)         # [B]

        w_hard = F.one_hot(idx, K).to(dtype)

        # soft surrogate: relative timing only
        any_k = s.any(dim=1)                  # [B,K]
        c = s.int().cumsum(dim=1)
        first_mask = (c == 1) & s
        t_first = first_mask.float().argmax(dim=1)             # [B,K]
        t_first = torch.where(any_k, t_first, torch.full_like(t_first, L))

        r = -t_first.to(dtype)
        w_sur = torch.softmax(r / self.temperature, dim=-1)

        w = w_hard.detach() - w_sur.detach() + w_sur
        spikes_gated = spikes * w.unsqueeze(1)
        return idx, w, spikes_gated


# ============================================================
# 5) Two-channel model: HankelFilterBank -> Trace -> MultiLIF -> WTA
# ============================================================
class TwoChannelHankelTraceLIF_MinWTA(nn.Module):
    """
    Returns per channel:
      (d2, z, spikes, series, v_seq, idx, w, spikes_gated, warmup)
    """
    def __init__(
        self,
        hankel_size: int,
        K: int,
        trace_lam=0.95,
        lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5,
        wta_temperature=0.2, wta_spike_thr=0.5,
    ):
        super().__init__()
        # different filterbanks for x and y (as you wanted)
        self.fbX = HankelFilterBank(hankel_size=hankel_size, K=K)
        self.fbY = HankelFilterBank(hankel_size=hankel_size, K=K)

        self.trX = PerFilterTrace(lam=trace_lam)
        self.trY = PerFilterTrace(lam=trace_lam)

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
# 6) Phase-1 loss: per-neuron variance of Δ = t_spike - t_onset (winner-only)
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
    min_count: int = 2,
):
    B, L, K = spikes.shape
    device = spikes.device
    dtype = v_seq.dtype
    b = torch.arange(B, device=device)

    # winner traces [B,L]
    s = spikes[b, :, idx]
    v = v_seq[b, :, idx]

    th_sp = float(lif_threshold)
    th_on = float(onset_frac) * float(lif_threshold)

    # ---- HARD times ----
    on_bool = (v >= th_on)
    sp_bool = (s > thr_spike)

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

    # ---- SOFT times ----
    t = torch.arange(L, device=device, dtype=dtype).unsqueeze(0)  # [1,L]

    a_on = torch.sigmoid(alpha * (v - th_on))
    p_on = a_on / (a_on.sum(dim=1, keepdim=True) + 1e-6)
    t_on_s = (p_on * t).sum(dim=1)

    a_sp = torch.sigmoid(alpha * (v - th_sp))
    p_sp = a_sp / (a_sp.sum(dim=1, keepdim=True) + 1e-6)
    t_sp_s = (p_sp * t).sum(dim=1)

    delta_s = t_sp_s - t_on_s

    # STE on delta
    delta = delta_h.detach() - delta_s.detach() + delta_s  # [B]

    # ---- per-neuron var over samples where idx==k ----
    loss = delta.new_zeros(())
    mean_term = delta.new_zeros(())
    for k in range(K):
        m = (idx == k)
        if int(m.sum().item()) >= min_count:
            dk = delta[m]
            loss = loss + dk.var(unbiased=False)
            mean_term = mean_term + dk.mean()

    return loss + lambda_mean * mean_term


# ============================================================
# 7) TRAINING LOOP (Phase 1)
# ============================================================
def train_phase1(
    steps=2000,
    B=32,
    L=400,
    hankel_size=20,
    K=12,
    lr=1e-3,
    device="cpu",
    onset_frac=0.6,
    alpha=10.0,
    lambda_mean=0.0,
    lambda_spike=1e-4,
):
    model = TwoChannelHankelTraceLIF_MinWTA(
        hankel_size=hankel_size,
        K=K,
        trace_lam=0.95,
        lif_tau=20.0, lif_th=1.5, lif_tau_adapt=100.0, lif_beta_adapt=1.5, lif_v_reset=-0.5,
        wta_temperature=0.2, wta_spike_thr=0.5,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        clean, noisy = make_sine_burst_batch(
            B=B, L=L,
            f1_range=(1.0, 3.0),
            f2_range=(6.0, 12.0),
            amp2=0.7,
            noise_std=0.12,
            burst_prob=0.9,
            burst_std=0.9,
            burst_len_range=(10, 60),
            device=device,
        )

        # x = noisy, y = clean (like your old setup)
        x = noisy
        y = clean

        (d2x, zx, sx, series_x, vx, idx_x, w_x, sx_g, warmup), \
        (d2y, zy, sy, series_y, vy, idx_y, w_y, sy_g, warmup2) = model(x, y)

        # phase-1: per-neuron jitter on winner latency, separately per channel
        loss_x = latency_jitter_loss_per_neuron(
            spikes=sx, v_seq=vx, idx=idx_x,
            lif_threshold=model.lifX.th,
            onset_frac=onset_frac, alpha=alpha,
            lambda_mean=lambda_mean
        )
        loss_y = latency_jitter_loss_per_neuron(
            spikes=sy, v_seq=vy, idx=idx_y,
            lif_threshold=model.lifY.th,
            onset_frac=onset_frac, alpha=alpha,
            lambda_mean=lambda_mean
        )

        # tiny anti-silence / activity shaping term (optional)
        spike_pen = (sx.mean() + sy.mean())

        loss = loss_x + loss_y + lambda_spike * spike_pen

        opt.zero_grad()
        loss.backward()
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
        lr=4e-4,
        device="cpu",
        onset_frac=0.6,
        alpha=10.0,
        lambda_mean=0.0,
        lambda_spike=1e-4,
    )
