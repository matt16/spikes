import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import matplotlib.pyplot as plt


# =========================
# DATA
# =========================
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
    return (returns_1d.flatten() < thr).float()


# =========================
# WINDOWS
# =========================
def make_windows(x_1d: torch.Tensor, y_1d: torch.Tensor, W: int, stride: int = 1):
    """
    x_1d: [T], y_1d: [T]
    returns:
      X: [N,1,W], Y: [N,W]
    """
    x = x_1d.flatten().unfold(0, W, stride)  # [N,W]
    y = y_1d.flatten().unfold(0, W, stride)  # [N,W]
    return x.unsqueeze(1).contiguous(), y.contiguous()


# =========================
# SURROGATE SPIKE
# =========================
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


# =========================
# MODEL
# =========================
class MinimalConvWTA_LIF(nn.Module):
    def __init__(
        self,
        kernels=(8, 16, 32),
        alpha=0.95,
        theta=0.05,
        gamma=12.0,
        gain_init=1.0,
        gain_min=0.1,
        gain_max=50.0,
        # PI controller params kept, but controller is now external to forward()
        Kp=0.8,
        Ki=0.10,
        i_min=-10.0,
        i_max=10.0,
        v_clip=None,   # optional numerical stabilization
    ):
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        assert len(self.kernels) >= 1 and all(k >= 1 for k in self.kernels)

        self.convs = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=k, bias=False) for k in self.kernels])

        # same init logic as before
        for conv, k in zip(self.convs, self.kernels):
            nn.init.normal_(conv.weight, mean=0.0, std=1.0 / math.sqrt(k))

        self.alpha = float(alpha)
        self.theta = float(theta)
        self.gamma = float(gamma)

        self.register_buffer("gain", torch.tensor(float(gain_init)), persistent=True)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)

        self.register_buffer("i_state", torch.tensor(0.0), persistent=True)
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.i_min = float(i_min)
        self.i_max = float(i_max)

        self.v_clip = v_clip

    def _fix_x_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"Expected x shape [B,1,T] (or [B,T]), got {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor):
        x = self._fix_x_shape(x)

        B, _, T = x.shape
        K = len(self.kernels)
        device, dtype = x.device, x.dtype

        # conv bank (same as before, causal)
        u_list = []
        for conv, k in zip(self.convs, self.kernels):
            x_pad = F.pad(x, (k - 1, 0))
            ui = conv(x_pad)[:, 0, :]             # [B,T]
            u_list.append(ui)
        u = torch.stack(u_list, dim=1)            # [B,K,T]

        # global gain
        g = float(self.gain.item())
        u = u * g

        # LIF bank with WTA
        v = torch.zeros((B, K), device=device, dtype=dtype)
        s_all = torch.zeros((B, K, T), device=device, dtype=dtype)

        for t in range(T):
            v = self.alpha * v + u[:, :, t]

            if self.v_clip is not None:
                v = torch.clamp(v, -float(self.v_clip), float(self.v_clip))

            m = v - self.theta
            s_raw = SurrogateSpike.apply(m, self.gamma)

            k_star = m.argmax(dim=1)  # [B]
            wta_mask = F.one_hot(k_star, num_classes=K).to(device=device, dtype=dtype)
            s = s_raw * wta_mask

            v = v - s * self.theta
            s_all[:, :, t] = s

        return u, s_all

    @torch.no_grad()
    def update_gain_controller(self, spikes: torch.Tensor, y: torch.Tensor):
        """
        External PI controller update.
        spikes: [B,K,T]
        y: [B,T] or [B,1,T]
        """
        if y.dim() == 3 and y.size(1) == 1:
            y = y[:, 0, :]
        if y.dim() != 2:
            raise ValueError(f"Expected y shape [B,T], got {tuple(y.shape)}")

        B, K, T = spikes.shape
        g = float(self.gain.item())

        total_spikes = float(spikes.sum().item())
        total_events = float(y.sum().item())

        spike_rate = total_spikes / (B * K * T + 1e-6)
        event_rate = total_events / (B * T + 1e-6)

        e = event_rate - spike_rate

        at_max = (g >= self.gain_max - 1e-12)
        at_min = (g <= self.gain_min + 1e-12)

        integrate = True
        if (at_max and e > 0) or (at_min and e < 0):
            integrate = False

        if integrate:
            i_new = float(self.i_state.item()) + e
            i_new = max(self.i_min, min(self.i_max, i_new))
            self.i_state.copy_(torch.tensor(i_new, dtype=self.i_state.dtype, device=self.i_state.device))

        i_val = float(self.i_state.item())

        z = math.log(max(g, 1e-12)) + (self.Kp * e + self.Ki * i_val)
        g_new = math.exp(z)
        g_new = max(self.gain_min, min(self.gain_max, g_new))
        self.gain.copy_(torch.tensor(g_new, dtype=self.gain.dtype, device=self.gain.device))

        return {
            "spike_rate": spike_rate,
            "event_rate": event_rate,
            "error": e,
            "gain": g_new,
            "i_state": float(self.i_state.item()),
        }


# =========================
# LOSS
# =========================
def nearest_event_distance_loss(spikes: torch.Tensor, y: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Original loss kept for now.
    spikes: [B,K,T]
    y: [B,T]
    """
    if y.dim() == 3 and y.size(1) == 1:
        y = y[:, 0, :]

    B, K, T = spikes.shape
    device, dtype = spikes.device, spikes.dtype

    d = torch.zeros((B, T), device=device, dtype=dtype)
    t_idx = torch.arange(T, device=device)

    for b in range(B):
        ev = torch.where(y[b] > 0.5)[0]
        if ev.numel() == 0:
            d[b].zero_()
        else:
            dist = (t_idx[:, None] - ev[None, :]).abs().to(dtype)
            d[b] = dist.min(dim=1).values

    spike_sum = spikes.sum(dim=(1, 2))
    weighted = (spikes * d[:, None, :]).sum(dim=(1, 2))
    per_sample = weighted / (spike_sum + eps)
    return per_sample.mean()


# =========================
# GAIN CALIBRATION
# =========================
@torch.no_grad()
def calibrate_gain(
    model,
    X,
    Y,
    steps=30,
    bs=256,
    device="cpu",
    verbose=True,
):
    """
    Calibrate global gain before weight training.
    Uses the existing PI controller logic, but outside forward().
    No gradient updates here.
    """
    model.eval()
    model.to(device)

    N = X.size(0)
    hist = {
        "gain": [],
        "spike_rate": [],
        "event_rate": [],
        "error": [],
        "i_state": [],
    }

    for step in range(1, steps + 1):
        if bs >= N:
            j = torch.arange(N)
        else:
            j = torch.randint(0, N, (bs,))

        x = X[j].to(device)
        y = Y[j].to(device).float()

        _, s = model(x)
        info = model.update_gain_controller(s, y)

        hist["gain"].append(info["gain"])
        hist["spike_rate"].append(info["spike_rate"])
        hist["event_rate"].append(info["event_rate"])
        hist["error"].append(info["error"])
        hist["i_state"].append(info["i_state"])

        if verbose:
            print(
                f"Calib {step:03d} | "
                f"spike_rate {info['spike_rate']:.5f} | "
                f"event_rate {info['event_rate']:.5f} | "
                f"err {info['error']:.5f} | "
                f"gain {info['gain']:.5f} | "
                f"i_state {info['i_state']:.5f}"
            )

    return hist


# =========================
# TRAIN
# =========================
def train(model, Xtr, Ytr, Xte, Yte, epochs=20, bs=64, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train": [], "test": []}

    def run_epoch(X, Y, train_mode: bool):
        model.train() if train_mode else model.eval()
        idx = torch.randperm(X.size(0), device="cpu") if train_mode else torch.arange(X.size(0))
        total, n = 0.0, 0

        for i in range(0, X.size(0), bs):
            j = idx[i:i + bs]
            x = X[j].to(device)
            y = Y[j].to(device).float()

            if train_mode:
                opt.zero_grad(set_to_none=True)

            _, s = model(x)
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

        print(
            f"Epoch {ep:03d} | train {tr:.5f} | test {te:.5f} | "
            f"fixed_gain {float(model.gain.item()):.5f} | "
            f"i_state {float(model.i_state.item()):.5f}"
        )

    return hist


# =========================
# PLOTS
# =========================
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
def plot_gain_calibration(calib_hist):
    plt.figure()
    plt.plot(calib_hist["gain"], label="gain")
    plt.plot(calib_hist["spike_rate"], label="spike_rate")
    plt.plot(calib_hist["event_rate"], label="event_rate")
    plt.xlabel("calibration step")
    plt.ylabel("value")
    plt.title("Gain calibration")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(calib_hist["error"], label="error = event_rate - spike_rate")
    plt.plot(calib_hist["i_state"], label="i_state")
    plt.xlabel("calibration step")
    plt.ylabel("value")
    plt.title("PI controller state during gain calibration")
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

    _, s = model(x)
    s = s[0].cpu()  # [K,T]
    y = y[0].cpu()  # [T]
    K, T = s.shape

    plt.figure()
    ev = torch.where(y > 0.5)[0].numpy()
    if len(ev) > 0:
        plt.scatter(ev, [K + 1] * len(ev), marker="x", label="TARGET")

    for k in range(K):
        sp = torch.where(s[k] > 0.5)[0].numpy()
        if len(sp) > 0:
            plt.scatter(sp, [k] * len(sp), marker="|")

    plt.yticks(list(range(K)) + [K + 1], [f"k={model.kernels[i]}" for i in range(K)] + ["TARGET"])
    plt.xlabel("t within window")
    plt.title("WTA spikes vs target events (one window)")
    plt.legend()
    plt.show()


@torch.no_grad()
def plot_input_and_spikes(model, X, Y, idx=None, device="cpu"):
    model.eval()
    model.to(device)
    if idx is None:
        idx = X.size(0) // 2

    x = X[idx:idx + 1].to(device)
    y = Y[idx:idx + 1].to(device).float()

    _, s = model(x)

    x_ = x[0, 0].cpu()
    y_ = y[0].cpu()
    s_any = s[0].sum(dim=0).cpu()  # [T]

    plt.figure(figsize=(12, 4))
    plt.plot(x_.numpy(), label="weekly return")
    ev = torch.where(y_ > 0.5)[0].numpy()
    sp = torch.where(s_any > 0.5)[0].numpy()

    if len(ev) > 0:
        plt.scatter(ev, x_[ev].numpy(), marker="x", label="TARGET")
    if len(sp) > 0:
        plt.scatter(sp, x_[sp].numpy(), marker="o", label="SPIKE(any kernel)")

    plt.xlabel("t within window")
    plt.title("Input returns + target events + spikes")
    plt.legend()
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    r = load_sp500_weekly_returns(start="1950-01-01", log_returns=False)  # [T]
    y = make_labels(r, thr=-0.02)                                         # [T]

    split = int(0.8 * r.numel())
    r_tr, r_te = r[:split], r[split:]
    y_tr, y_te = y[:split], y[split:]

    W = 256
    Xtr, Ytr = make_windows(r_tr, y_tr, W=W, stride=1)
    Xte, Yte = make_windows(r_te, y_te, W=W, stride=1)

    model = MinimalConvWTA_LIF(
        kernels=(8, 16, 32),
        alpha=0.95,
        theta=0.05,
        gamma=12.0,
        gain_init=1.0,
        gain_min=0.1,
        gain_max=50.0,
        Kp=0.8,
        Ki=0.10,
        i_min=-10.0,
        i_max=10.0,
        v_clip=None,   # set e.g. 5.0 if you want clipping
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("\n=== Gain calibration before training ===")
    calib_hist = calibrate_gain(
        model,
        Xtr,
        Ytr,
        steps=30,
        bs=256,
        device=device,
        verbose=True,
    )

    print("\n=== Weight training with fixed calibrated gain ===")
    hist = train(model, Xtr, Ytr, Xte, Yte, epochs=20, bs=64, lr=1e-3, device=device)

    plot_gain_calibration(calib_hist)
    plot_loss(hist)
    plot_spikes_vs_events(model, Xte, Yte, idx=Xte.size(0) // 2, device=device)
    plot_input_and_spikes(model, Xte, Yte, idx=Xte.size(0) // 2, device=device)