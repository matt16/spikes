import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import matplotlib.pyplot as plt


# ── Data ─────────────────────────────────────────────────────────────────────

def load_sp500_weekly(start="1950-01-01", log_returns=False) -> torch.Tensor:
    df = yf.download("^GSPC", start=start, progress=False, auto_adjust=False)
    price = df["Adj Close"].resample("W-FRI").last().dropna()
    r = np.log(price / price.shift(1)) if log_returns else price.pct_change()
    return torch.from_numpy(r.dropna().to_numpy(np.float32))

def make_labels(r: torch.Tensor, thr: float = -0.02) -> torch.Tensor:
    return (r < thr).float()

def make_windows(x: torch.Tensor, y: torch.Tensor, W: int, stride: int = 1):
    """Returns X [N,1,W] and Y [N,W]."""
    X = x.unfold(0, W, stride).unsqueeze(1).contiguous()   # [N,1,W]
    Y = y.unfold(0, W, stride).contiguous()                # [N,W]
    return X, Y


# ── Surrogate spike ───────────────────────────────────────────────────────────
# Forward : hard Heaviside
# Backward: triangular surrogate gradient

class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_theta: torch.Tensor, gamma: float):
        ctx.save_for_backward(v_minus_theta)
        ctx.gamma = gamma
        return (v_minus_theta >= 0).to(v_minus_theta.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        g = ctx.gamma
        # triangular window: (1 - g|x|)+ · g
        surrogate = g * (1.0 - g * x.abs()).clamp(min=0.0)
        return grad * surrogate, None

spike = _Spike.apply


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvLIF_WTA(nn.Module):
    """
    Single Conv1d with K same-size filters → K LIF neurons → WTA.

    LIF ODE (Euler, dt=1):
        v[t] = alpha * v[t-1] + (1-alpha) * u[t]     # leak
        s[t] = H(v[t] - theta)                        # spike (hard / surrogate)
        v[t] = v[t] - s[t] * theta                    # reset

    where alpha = exp(-1/tau)  comes directly from the continuous LIF equation.

    WTA rule: only the channel with the highest membrane fires,
              and only if that winner actually crossed theta.
    """

    def __init__(
        self,
        K: int        = 8,
        kernel_size: int = 16,
        tau: float    = 10.0,   # membrane time constant (steps)
        theta: float  = 0.5,    # firing threshold
        gamma: float  = 10.0,   # surrogate sharpness
    ):
        super().__init__()
        self.K           = K
        self.kernel_size = kernel_size
        self.theta       = theta
        self.gamma       = gamma
        self.alpha       = math.exp(-1.0 / tau)   # from LIF ODE

        # K filters, all same length, shared input channel
        self.conv = nn.Conv1d(1, K, kernel_size=kernel_size, bias=False)
        nn.init.normal_(self.conv.weight, 0.0, 1.0 / math.sqrt(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, T]
        returns spikes : [B, K, T]
        """
        x = x.reshape(x.shape[0], 1, -1)   # → [B, 1, T] regardless of input shape
        B, _, T = x.shape
        alpha = self.alpha
        beta  = 1.0 - alpha

        # ── causal convolution ────────────────────────────────────────────
        u = self.conv(F.pad(x, (self.kernel_size - 1, 0)))  # [B, K, T]

        # ── LIF + WTA loop (over time) ────────────────────────────────────
        v      = x.new_zeros(B, self.K)
        spikes = x.new_zeros(B, self.K, T)

        for t in range(T):
            # LIF ODE step
            v = alpha * v + beta * u[:, :, t]           # [B, K]

            # Hard spike in forward, triangular surrogate in backward
            s_raw = spike(v - self.theta, self.gamma)   # [B, K]

            # WTA: winner = channel with highest membrane
            winner        = v.argmax(dim=1)             # [B]
            winner_spiked = s_raw.gather(1, winner.unsqueeze(1)).squeeze(1)  # [B]
            # fire only if winner actually spiked
            wta_mask = F.one_hot(winner, self.K).to(x.dtype) * winner_spiked.unsqueeze(1)

            s = s_raw * wta_mask                        # [B, K]
            v = v - s * self.theta                      # soft reset

            spikes[:, :, t] = s

        return spikes   # [B, K, T]


# ── Loss ──────────────────────────────────────────────────────────────────────

def latency_loss(spikes: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    For every predicted spike find the nearest target event → mean latency.

    Vectorised: precompute d[b,t] = distance to nearest target (no grad needed),
    then weight each soft spike by its distance.

        L = sum_t( pred[b,t] · d[b,t] ) / ( sum_t pred[b,t] + eps )

    Gradients flow through pred (= sum of surrogate spikes across K channels).
    """
    B, K, T = spikes.shape
    y = y.reshape(B, T)   # handle [B,1,T] or [B,T] or any extra dims
    t_idx = torch.arange(T, device=spikes.device, dtype=spikes.dtype)

    # ── distance map (no grad, fully vectorised) ─────────────────────────
    with torch.no_grad():
        ev_mask = (y > 0.5).float()                             # [B, T]
        INF     = float(T)

        # pairwise |t_query - t_event| → [T, T], then expand to [B, T, T]
        raw_dist = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).abs().float()  # [T, T]
        raw_dist = raw_dist.unsqueeze(0).expand(B, T, T)       # [B, T, T]

        # set non-event columns to INF via broadcast mask [B, 1, T] → [B, T, T]
        col_mask = ev_mask.unsqueeze(1).expand(B, T, T)        # [B, T, T]
        dist_inf = raw_dist * col_mask + INF * (1.0 - col_mask)

        d = dist_inf.min(dim=2).values                         # [B, T]
        # samples without any event: no penalty
        d = d.masked_fill(ev_mask.sum(dim=1, keepdim=True) == 0, 0.0)

    # ── weighted mean latency ─────────────────────────────────────────────
    pred = spikes.sum(dim=1)                                # [B, T], soft (has grad)
    weighted = (pred * d).sum(dim=1)                        # [B]
    total    = pred.sum(dim=1) + eps                        # [B]
    return (weighted / total).mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, Xtr, Ytr, Xte, Yte, epochs=20, bs=64, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"train": [], "test": []}

    for ep in range(1, epochs + 1):
        for split, (X, Y, train_mode) in enumerate(
            [(Xtr, Ytr, True), (Xte, Yte, False)]
        ):
            model.train(train_mode)
            idx   = torch.randperm(X.size(0)) if train_mode else torch.arange(X.size(0))
            total = 0.0

            for i in range(0, X.size(0), bs):
                j = idx[i : i + bs]
                x, y = X[j].to(device), Y[j].to(device)

                if train_mode:
                    opt.zero_grad(set_to_none=True)

                s    = model(x)
                loss = latency_loss(s, y)

                if train_mode:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                total += loss.item() * x.size(0)

            avg = total / X.size(0)
            key = "train" if split == 0 else "test"
            hist[key].append(avg)

        print(f"Epoch {ep:03d} | train {hist['train'][-1]:.4f} | test {hist['test'][-1]:.4f}")

    return hist


# ── Plots ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_loss(hist):
    plt.figure()
    plt.plot(hist["train"], label="train")
    plt.plot(hist["test"],  label="test")
    plt.xlabel("epoch"); plt.ylabel("latency loss")
    plt.title("Training history"); plt.legend(); plt.tight_layout(); plt.show()


@torch.no_grad()
def plot_spikes(model, X, Y, idx=None, device="cpu"):
    model.eval().to(device)
    idx = idx if idx is not None else X.size(0) // 2

    x = X[idx : idx + 1].to(device)
    y = Y[idx : idx + 1].to(device)

    s = model(x)[0].cpu()  # [K, T]
    y = y[0].cpu()          # [T]
    K, T = s.shape

    fig, ax = plt.subplots(figsize=(12, 3))
    ev = torch.where(y > 0.5)[0].numpy()
    ax.scatter(ev, np.full(len(ev), K + 0.5), marker="x", s=60, color="red", label="target", zorder=3)

    for k in range(K):
        sp = torch.where(s[k] > 0.5)[0].numpy()
        ax.scatter(sp, np.full(len(sp), k), marker="|", s=50)

    ax.set_yticks(list(range(K)) + [K + 0.5])
    ax.set_yticklabels([f"filter {k}" for k in range(K)] + ["TARGET"])
    ax.set_xlabel("time step"); ax.set_title("WTA spikes vs target events")
    ax.legend(); plt.tight_layout(); plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    r = load_sp500_weekly(start="1950-01-01", log_returns=False)
    y = make_labels(r, thr=-0.02)

    split      = int(0.8 * len(r))
    Xtr, Ytr   = make_windows(r[:split], y[:split], W=64)
    Xte, Yte   = make_windows(r[split:], y[split:], W=64)

    model = ConvLIF_WTA(
        K           = 8,
        kernel_size = 16,
        tau         = 10.0,
        theta       = 0.5,
        gamma       = 10.0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Train windows: {len(Xtr)} | Test windows: {len(Xte)}")

    hist = train(model, Xtr, Ytr, Xte, Yte, epochs=20, bs=64, lr=1e-3, device=device)
    plot_loss(hist)
    plot_spikes(model, Xte, Yte, device=device)
