import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# 0) Synthetic data: TWO sinusoids + constant Gaussian noise
#    Frequencies are specified as "cycles per training window"
# ============================================================
def make_sine_long_constant_noise(
    T: int,
    L_win: int = 400,
    f1_cycles_per_win=(0.7, 1.8),
    f2_cycles_per_win=(2.5, 6.0),
    amp2=0.7,
    noise_std=0.05,
    device="cpu",
    generator: torch.Generator | None = None,
):
    """
    Long signal with 2 sinusoids. Frequencies are drawn in cycles/window and
    converted to cycles/sample so that within a window of length L_win you
    reliably see oscillations.

    If you pass a torch.Generator with a fixed seed, the produced long sequence
    is deterministic.
    """
    n = torch.arange(T, device=device, dtype=torch.float32)  # [T] samples

    # draw cycles/window then convert to cycles/sample
    c1 = torch.empty((), device=device).uniform_(*f1_cycles_per_win, generator=generator)
    c2 = torch.empty((), device=device).uniform_(*f2_cycles_per_win, generator=generator)
    f1 = c1 / float(L_win)
    f2 = c2 / float(L_win)

    p1 = torch.empty((), device=device).uniform_(0, 2 * math.pi, generator=generator)
    p2 = torch.empty((), device=device).uniform_(0, 2 * math.pi, generator=generator)

    clean = torch.sin(2 * math.pi * f1 * n + p1) + amp2 * torch.sin(2 * math.pi * f2 * n + p2)

    noise = torch.randn(
        clean.shape,
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )

    noisy = clean + noise_std * noise
    return clean, noisy  # [T], [T]


# ============================================================
# Optional plot: long sequence with one sampled window highlighted
# ============================================================
def plot_window_in_context(noisy_long, L, context=2000, start=None):
    """
    Plot long sequence with one training window highlighted.
    If start is None: chooses a random start.
    """
    T_long = noisy_long.shape[0]
    if start is None:
        start = torch.randint(0, T_long - L + 1, (1,)).item()

    c0 = max(0, start - context)
    c1 = min(T_long, start + L + context)

    plt.figure(figsize=(12, 3))
    plt.plot(range(c0, c1), noisy_long[c0:c1].cpu(), alpha=0.7, label="long sequence")
    plt.axvspan(start, start + L, color="red", alpha=0.25, label="training window")
    plt.xlabel("absolute time index")
    plt.ylabel("signal")
    plt.title("Random training window inside long sequence")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 1) Conv Filter Bank (causal)
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
        x1 = x.unsqueeze(1)  # [B,1,L]
        outs = []
        for conv, k in zip(self.branches, self._unique_sizes):
            outs.append(conv(F.pad(x1, (k - 1, 0))))  # [B,Ck,L]
        d2 = torch.cat(outs, 1).transpose(1, 2)  # [B,L,K]
        warmup = self.kmax - 1
        if warmup > 0:
            d2 = d2[:, warmup:, :]
        return d2, warmup


# ============================================================
# 3) MultiLIF (hard forward spikes, surrogate gradients)
# ============================================================
class MultiLIF(nn.Module):
    def __init__(
        self,
        tau=20.0,
        threshold=1.5,
        tau_adapt=100.0,
        beta_adapt=1.5,
        v_reset=-0.5,
        surr_slope=10.0,
    ):
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
        v_seq = torch.zeros(B, L, K, device=I.device, dtype=I.dtype)
        k_surr = self.surr_slope

        for t in range(L):
            th_eff = self.th + self.beta_adapt * a
            v = v + (-v / self.tau + I[:, t, :])
            v_seq[:, t, :] = v

            s_soft = torch.sigmoid(k_surr * (v - th_eff))
            s_hard = (v >= th_eff).to(I.dtype)
            s = s_hard.detach() - s_soft.detach() + s_soft  # forward hard, backward soft

            spikes[:, t, :] = s

            v = v * (1 - s.detach()) + self.v_reset * s.detach()
            a = a + (-a / self.tau_adapt + s)

        return spikes, v_seq


# ============================================================
# 4) One-shot silencing: keep first HARD spike only per (B,K)
# ============================================================
class OneShotSilencer(nn.Module):
    def forward(self, spikes):  # [B,L,K]
        hard = (spikes > 0).to(spikes.dtype)
        c = hard.cumsum(1)
        keep = ((c == 1) & (hard > 0)).to(spikes.dtype)
        return spikes * keep


# ============================================================
# 5) Loss: HARD forward delta + Huber scatter, gradients via soft time
# ============================================================
def latency_jitter_huber_hard_forward(
    spikes: torch.Tensor,   # [B,L,K]
    v_seq: torch.Tensor,    # [B,L,K]
    lif_threshold: float,
    onset_frac: float = 0.6,
    alpha: float = 10.0,
    min_count: int = 2,
    huber_beta: float = 10.0,
):
    B, L, K = spikes.shape
    th_sp = float(lif_threshold)
    th_on = float(onset_frac) * th_sp

    loss = spikes.new_zeros(())
    t = torch.arange(L, device=spikes.device, dtype=v_seq.dtype).unsqueeze(0)  # [1,L]

    for k in range(K):
        s = spikes[:, :, k]
        v = v_seq[:, :, k]

        on = (v >= th_on)
        sp = (s > 0)

        # HARD first onset/spike times (forward)
        t_on_h = ((on.int().cumsum(1) == 1) & on).float().argmax(1)
        t_on_h = torch.where(on.any(1), t_on_h, torch.full_like(t_on_h, L))

        t_sp_h = ((sp.int().cumsum(1) == 1) & sp).float().argmax(1)
        t_sp_h = torch.where(sp.any(1), t_sp_h, torch.full_like(t_sp_h, L))

        delta_h = (t_sp_h.clamp(0, L - 1) - t_on_h.clamp(0, L - 1)).to(v.dtype)

        # SOFT surrogate times (for gradients)
        a_on = torch.sigmoid(alpha * (v - th_on))
        a_sp = torch.sigmoid(alpha * (v - th_sp))
        t_on_s = (a_on / (a_on.sum(1, keepdim=True) + 1e-6) * t).sum(1)
        t_sp_s = (a_sp / (a_sp.sum(1, keepdim=True) + 1e-6) * t).sum(1)
        delta_s = (t_sp_s - t_on_s)

        # ST: forward hard, backward soft
        delta = delta_h.detach() - delta_s.detach() + delta_s

        m = on.any(1) & sp.any(1)
        if int(m.sum()) >= min_count:
            d = delta[m]
            mu = d.mean()
            loss = loss + F.smooth_l1_loss(d, mu.expand_as(d), beta=huber_beta)

    return loss


# ============================================================
# Model: Conv -> LIF -> OneShot
# ============================================================
class SimpleConvLIF(nn.Module):
    def __init__(self, kernel_sizes=(3, 5, 9, 17), lif_kwargs=None):
        super().__init__()
        self.conv = ConvFilterBank(kernel_sizes)
        self.lif = MultiLIF(**(lif_kwargs or {}))
        self.oneshot = OneShotSilencer()

    def forward(self, x):  # [B,L]
        d2, warmup = self.conv(x)
        spikes, v_seq = self.lif(d2)
        spikes = self.oneshot(spikes)
        return spikes, v_seq, warmup


# ============================================================
# Training
# ============================================================
def train_phase1(
    steps=500,
    B=128,
    L=400,
    kernel_sizes=(3, 5, 9, 17),
    T_long=200_000,
    lr=3e-5,
    grad_clip_norm=1.0,
    plot_example_window=True,
    window_context=2000,
    seed_long=0,            # <<< NEW: fixes the long sequence shape
    seed_windows=None,      # optional: fix window sampling too
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # --- Generator for long sequence (fixed!) ---
    gen_long = torch.Generator(device=device)
    gen_long.manual_seed(int(seed_long))

    _, noisy_long = make_sine_long_constant_noise(
        T_long,
        L_win=L,
        device=device,
        generator=gen_long,   # <<< makes long sequence deterministic
    )

    if plot_example_window:
        plot_window_in_context(noisy_long, L=L, context=window_context)

    model = SimpleConvLIF(kernel_sizes=kernel_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Optional generator for window sampling ---
    gen_win = None
    if seed_windows is not None:
        gen_win = torch.Generator(device=device)
        gen_win.manual_seed(int(seed_windows))

    loss_hist = []

    for step in range(steps):
        starts = torch.randint(
            0, T_long - L + 1, (B,),
            device=device,
            generator=gen_win,   # None => random each run; set seed_windows => deterministic
        )
        idx_t = starts[:, None] + torch.arange(L, device=device)[None, :]
        x = noisy_long[idx_t]  # [B,L]

        spikes, v_seq, warmup = model(x)

        loss = latency_jitter_huber_hard_forward(
            spikes=spikes,
            v_seq=v_seq,
            lif_threshold=model.lif.th,
            onset_frac=0.6,
            alpha=10.0,
            min_count=2,
            huber_beta=10.0,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

        opt.step()

        loss_hist.append(float(loss.detach().cpu()))

        if step % 25 == 0 or step == steps - 1:
            spike_mean = float(spikes.mean().detach().cpu())
            fired_frac = float((spikes.sum(1) > 0).float().mean().detach().cpu())
            print(f"{step:4d} | loss={loss_hist[-1]:.6f} | spike_mean={spike_mean:.6f} | fired_frac={fired_frac:.3f}")

    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("Hard-forward latency jitter (Huber) over training")
    plt.tight_layout()
    plt.show()

    return model, loss_hist


if __name__ == "__main__":
    train_phase1(
        steps=500,
        B=128,
        L=400,
        kernel_sizes=(3, 5, 9, 17),
        grad_clip_norm=1.0,
        plot_example_window=True,
        window_context=2000,
        seed_long=0,         # long sequence always same
        seed_windows=None,   # keep windows random; set e.g. 123 to freeze windows too
    )
