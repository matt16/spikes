import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import matplotlib.pyplot as plt


# -------------------------
# DATA (minimal)
# -------------------------
def load_sp500_weekly_returns(start="1950-01-01", end=None, log_returns=False) -> torch.Tensor:
    df = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise ValueError("No data from yfinance.")

    price = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    weekly = price.resample("W-FRI").last().dropna()

    rets = np.log(weekly / weekly.shift(1)) if log_returns else weekly.pct_change()
    rets = rets.dropna().to_numpy(dtype=np.float32)
    return torch.from_numpy(rets).flatten()  # [T]


def make_labels(returns_1d: torch.Tensor, thr=-0.02) -> torch.Tensor:
    returns_1d = returns_1d.flatten()
    return (returns_1d < thr).float()  # [T]


# -------------------------
# WINDOWS (no Dataset class)
# -------------------------
def make_windows(x_1d: torch.Tensor, y_1d: torch.Tensor, W: int, stride: int = 1, normalize: bool = True):
    """
    x_1d: [T], y_1d: [T]
    returns:
      X: [N,1,W], Y: [N,W]
    """
    x_1d = x_1d.flatten()
    y_1d = y_1d.flatten()
    if x_1d.numel() < W:
        raise ValueError(f"Time series too short for window W={W}: T={x_1d.numel()}")

    x = x_1d.unfold(0, W, stride)  # [N,W]
    y = y_1d.unfold(0, W, stride)  # [N,W]

    if normalize:
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        x = (x - mu) / sd

    return x.unsqueeze(1).contiguous(), y.contiguous()  # X [N,1,W], Y [N,W]


# -------------------------
# SURROGATE SPIKE (hard forward, "hard-ish" backward)
# -------------------------
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma: float):
        ctx.save_for_backward(x)
        ctx.gamma = gamma
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        gamma = ctx.gamma
        ax = (gamma * x).abs()
        grad = (ax < 1.0).to(x.dtype) * (1.0 - ax) * gamma  # triangular window
        return grad_output * grad, None


# -------------------------
# MODEL: conv-per-kernel -> global LIF bank with WTA + global gain control
# -------------------------
class MinimalConvWTA_LIF(nn.Module):
    def __init__(
        self,
        kernels=(8, 16, 32),
        alpha=0.95,
        theta=1.0,
        gamma=12.0,
        gain_init=1.0,
        gain_lr=0.20,
        gain_min=0.1,
        gain_max=10.0,
    ):
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        assert len(self.kernels) >= 1 and all(k >= 1 for k in self.kernels)

        self.convs = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=k, bias=False) for k in self.kernels])

        self.register_buffer(
            "scales",
            torch.tensor([1.0 / math.sqrt(k) for k in self.kernels], dtype=torch.float32),
        )

        for conv, k in zip(self.convs, self.kernels):
            nn.init.normal_(conv.weight, mean=0.0, std=1.0 / math.sqrt(k))

        self.alpha = float(alpha)
        self.theta = float(theta)
        self.gamma = float(gamma)

        # global gain (buffer, not a learnable parameter)
        self.register_buffer("gain", torch.tensor(float(gain_init)), persistent=True)
        self.gain_lr = float(gain_lr)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)

    def _fix_x_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          [B,1,T] (ok)
          [B,T]   -> becomes [B,1,T]
          [B,1,1,T] -> squeeze to [B,1,T]
        """
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"Expected x shape [B,1,T] (or [B,T]), got {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: [B,1,T] (or [B,T] / [B,1,1,T])
        y: [B,T] 0/1
        returns:
          u: [B,K,T]
          s: [B,K,T] hard spikes with WTA
        """
        x = self._fix_x_shape(x)
        if y.dim() == 3 and y.size(1) == 1:
            y = y[:, 0, :]
        if y.dim() != 2:
            raise ValueError(f"Expected y shape [B,T], got {tuple(y.shape)}")

        B, _, T = x.shape
        K = len(self.kernels)
        device, dtype = x.device, x.dtype

        # conv bank
        u_list = []
        scales = self.scales.to(device=device, dtype=dtype)
        for i, (conv, k) in enumerate(zip(self.convs, self.kernels)):
            x_pad = F.pad(x, (k - 1, 0))          # causal left pad
            ui = conv(x_pad)[:, 0, :]             # [B,T]
            ui = ui * scales[i]
            u_list.append(ui)
        u = torch.stack(u_list, dim=1)            # [B,K,T]

        # IMPORTANT FIX:
        # use gain as a Python float => not in the autograd graph => can be updated in-place safely
        g = float(self.gain.item())
        u = u * g

        # LIF bank
        v = torch.zeros((B, K), device=device, dtype=dtype)
        s_all = torch.zeros((B, K, T), device=device, dtype=dtype)

        for t in range(T):
            v = self.alpha * v + u[:, :, t]
            m = v - self.theta
            s_raw = SurrogateSpike.apply(m, self.gamma)  # hard forward

            # WTA: at most one spike per sample per t
            k_star = m.argmax(dim=1)
            wta_mask = F.one_hot(k_star, num_classes=K).to(dtype)
            s = s_raw * wta_mask

            v = v - s * self.theta
            s_all[:, :, t] = s

        # global gain control update (no grad; safe now)
        with torch.no_grad():
            total_spikes = s_all.sum()
            total_events = y.sum()

            spike_rate = total_spikes / (B * K * T + 1e-6)
            event_rate = total_events / (B * T + 1e-6)

            err = (event_rate - spike_rate)
            g_new = torch.tensor(g, device=device, dtype=dtype) * torch.exp(self.gain_lr * err)
            g_new = torch.clamp(g_new, self.gain_min, self.gain_max)
            self.gain.copy_(g_new.to(self.gain.dtype))

        return u, s_all


# -------------------------
# LOSS: nearest target event per spike
# -------------------------
def nearest_event_distance_loss(spikes: torch.Tensor, y: torch.Tensor, eps=1e-6) -> torch.Tensor:
    if y.dim() == 3 and y.size(1) == 1:
        y = y[:, 0, :]
    B, K, T = spikes.shape
    device, dtype = spikes.device, spikes.dtype

    d = torch.zeros((B, T), device=device, dtype=dtype)
    for b in range(B):
        ev = torch.where(y[b] > 0.5)[0]
        if ev.numel() == 0:
            d[b].zero_()
        else:
            t_idx = torch.arange(T, device=device)
            dist = (t_idx[:, None] - ev[None, :]).abs().to(dtype)
            d[b] = dist.min(dim=1).values

    spike_sum = spikes.sum(dim=(1, 2))
    weighted = (spikes * d[:, None, :]).sum(dim=(1, 2))
    per_sample = weighted / (spike_sum + eps)
    return per_sample.mean()


# -------------------------
# TRAIN
# -------------------------
def train(model, Xtr, Ytr, Xte, Yte, epochs=200, bs=64, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train": [], "test": []}

    def run_epoch(X, Y, train_mode: bool):
        model.train() if train_mode else model.eval()
        idx = torch.randperm(X.size(0)) if train_mode else torch.arange(X.size(0))
        total, n = 0.0, 0

        for i in range(0, X.size(0), bs):
            j = idx[i:i + bs]
            x = X[j].to(device)
            y = Y[j].to(device).float()

            if train_mode:
                opt.zero_grad(set_to_none=True)

            _, s = model(x, y)
            loss = nearest_event_distance_loss(s, y)

            if train_mode:
                loss.backward()
                opt.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

        return total / max(1, n)

    for ep in range(1, epochs + 1):
        tr = run_epoch(Xtr, Ytr, True)
        te = run_epoch(Xte, Yte, False)
        hist["train"].append(tr)
        hist["test"].append(te)
        print(f"Epoch {ep:03d} | train {tr:.5f} | test {te:.5f} | gain {float(model.gain.item()):.3f}")

    return hist


# -------------------------
# PLOTS
# -------------------------
@torch.no_grad()
def plot_loss(hist):
    plt.figure()
    plt.plot(hist["train"], label="train")
    plt.plot(hist["test"], label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss history")
    plt.legend()
    plt.show()


@torch.no_grad()
def plot_spikes_vs_events(model, X, Y, idx=None, device="cpu"):
    model.eval()
    model.to(device)
    if idx is None:
        idx = X.size(0) // 2

    x = X[idx:idx + 1].to(device)
    y = Y[idx:idx + 1].to(device).float()

    _, s = model(x, y)
    s = s[0].cpu()  # [K,T]
    y = y[0].cpu()  # [T]
    K, T = s.shape

    plt.figure()
    ev = torch.where(y > 0.5)[0].numpy()
    if len(ev) > 0:
        plt.scatter(ev, [K + 1] * len(ev), marker="x")

    for k in range(K):
        sp = torch.where(s[k] > 0.5)[0].numpy()
        if len(sp) > 0:
            plt.scatter(sp, [k] * len(sp), marker="|")

    plt.yticks(list(range(K)) + [K + 1], [f"k={model.kernels[i]}" for i in range(K)] + ["TARGET"])
    plt.xlabel("t within window")
    plt.title("WTA spikes vs target events (one window)")
    plt.legend()
    plt.show()


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    r = load_sp500_weekly_returns(start="1950-01-01", log_returns=False)  # [T]
    y = make_labels(r, thr=-0.02)                                         # [T]

    split = int(0.8 * r.numel())
    r_tr, r_te = r[:split], r[split:]
    y_tr, y_te = y[:split], y[split:]

    W = 256
    Xtr, Ytr = make_windows(r_tr, y_tr, W=W, stride=1, normalize=True)
    Xte, Yte = make_windows(r_te, y_te, W=W, stride=1, normalize=True)

    model = MinimalConvWTA_LIF(
        kernels=(8, 16, 32),
        alpha=0.95,
        theta=1.0,
        gamma=12.0,
        gain_init=1.0,
        gain_lr=0.20,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    hist = train(model, Xtr, Ytr, Xte, Yte, epochs=100, bs=64, lr=1e-3, device=device)
    plot_loss(hist)
    plot_spikes_vs_events(model, Xte, Yte, idx=Xte.size(0) // 2, device=device)