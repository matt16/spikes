import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# 0) Synthetic data: ONE long 2-sine series + constant Gaussian noise
# ============================================================
def make_sine_long_constant_noise(
    T: int,
    f1_range=(10.0, 20.0),
    f2_range=(30.0, 60.0),
    amp2=0.7,
    noise_std=0.05,
    device="cpu",
):
    t = torch.linspace(0.0, 1.0, T, device=device)

    f1 = torch.empty((), device=device).uniform_(*f1_range)
    f2 = torch.empty((), device=device).uniform_(*f2_range)
    p1 = torch.empty((), device=device).uniform_(0, 2 * math.pi)
    p2 = torch.empty((), device=device).uniform_(0, 2 * math.pi)

    clean = torch.sin(2 * math.pi * f1 * t + p1) + amp2 * torch.sin(2 * math.pi * f2 * t + p2)
    noisy = clean + noise_std * torch.randn_like(clean)
    return clean, noisy  # [T], [T]


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
# 3) MultiLIF (forward spikes are HARD 0/1; grads via surrogate)
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
            s = s_hard.detach() - s_soft.detach() + s_soft  # forward==hard, grads==soft

            spikes[:, t, :] = s
            s_num = s_num + s
            series[:, t, :] = s_num

            v = v * (1 - s.detach()) + self.v_reset * s.detach()
            a = a + (-a / self.tau_adapt + s)

        return spikes, series, v_seq


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
# 5) Loss: per-neuron latency jitter variance (FIX #1 applied)
#    IMPORTANT CHANGE: only count samples with onset AND spike
# ============================================================
def latency_jitter_loss_all_neurons(
    spikes: torch.Tensor,   # [B,L,K]
    v_seq: torch.Tensor,    # [B,L,K]
    lif_threshold: float,
    onset_frac: float = 0.6,
    alpha: float = 10.0,
    min_count: int = 2,
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

        # hard times
        t_on_h = ((on.int().cumsum(1) == 1) & on).float().argmax(1)
        t_on_h = torch.where(on.any(1), t_on_h, torch.full_like(t_on_h, L))

        t_sp_h = ((sp.int().cumsum(1) == 1) & sp).float().argmax(1)
        t_sp_h = torch.where(sp.any(1), t_sp_h, torch.full_like(t_sp_h, L))

        delta_h = (t_sp_h.clamp(0, L - 1) - t_on_h.clamp(0, L - 1)).to(v.dtype)

        # soft times for gradients
        a_on = torch.sigmoid(alpha * (v - th_on))
        a_sp = torch.sigmoid(alpha * (v - th_sp))
        t_on_s = (a_on / (a_on.sum(1, keepdim=True) + 1e-6) * t).sum(1)
        t_sp_s = (a_sp / (a_sp.sum(1, keepdim=True) + 1e-6) * t).sum(1)

        delta = delta_h.detach() - (t_sp_s - t_on_s).detach() + (t_sp_s - t_on_s)

        # FIX #1: only count samples where neuron had onset AND actually spiked
        m = on.any(1) & sp.any(1)

        if int(m.sum()) >= min_count:
            loss = loss + delta[m].var(unbiased=False)

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

    def forward(self, x):  # x: [B,L]
        d2, warmup = self.conv(x)
        spikes, _, v_seq = self.lif(d2)
        spikes = self.oneshot(spikes)
        return spikes, v_seq, warmup


# ============================================================
# Training + loss plot (RANDOM runs: no manual_seed)
# ============================================================
def train_phase1(
    steps=500,
    B=128,
    L=400,
    kernel_sizes=(3, 5, 9, 17),
    T_long=200_000,
    lr=3e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    _, noisy_long = make_sine_long_constant_noise(T_long, device=device)
    model = SimpleConvLIF(kernel_sizes=kernel_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_hist = []

    for step in range(steps):
        starts = torch.randint(0, T_long - L + 1, (B,), device=device)
        idx_t = starts[:, None] + torch.arange(L, device=device)[None, :]
        x = noisy_long[idx_t]  # [B,L]

        spikes, v_seq, warmup = model(x)

        loss = latency_jitter_loss_all_neurons(
            spikes=spikes,
            v_seq=v_seq,
            lif_threshold=model.lif.th,
            onset_frac=0.6,
            alpha=10.0,
            min_count=2,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    plt.title("Latency jitter loss over training")
    plt.show()

    return model, loss_hist


if __name__ == "__main__":
    train_phase1(
        steps=500,
        B=128,
        L=400,
        kernel_sizes=(3, 5, 9, 17),
    )
