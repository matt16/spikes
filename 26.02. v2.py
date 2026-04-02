import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Sequence, List, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# DATA
# =========================
def _get_price_series(df: pd.DataFrame) -> pd.Series:
    """Robust: holt Preisserie aus yfinance-DF, egal ob MultiIndex / Adj Close / Close."""
    if isinstance(df.columns, pd.MultiIndex):
        fields = list(df.columns.get_level_values(0).unique())
        if "Adj Close" in fields:
            s = df.xs("Adj Close", level=0, axis=1)
        elif "Close" in fields:
            s = df.xs("Close", level=0, axis=1)
        else:
            raise KeyError(f"No 'Adj Close' or 'Close' in MultiIndex columns: {fields}")
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s

    if "Adj Close" in df.columns:
        return df["Adj Close"]
    if "Close" in df.columns:
        return df["Close"]

    raise KeyError(f"No 'Adj Close' or 'Close' in columns: {list(df.columns)}")


def load_sp500_weekly_returns(
    start: str = "1950-01-01",
    end: str | None = None,
    use_log_returns: bool = False,
):
    """
    Lädt S&P500 (^GSPC) von Yahoo Finance, resampled auf Weekly (Freitag).
    """
    ticker = "^GSPC"
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if df is None or len(df) == 0:
        raise ValueError("No data downloaded from Yahoo Finance.")

    price = _get_price_series(df).dropna()
    weekly = price.resample("W-FRI").last().dropna()

    if use_log_returns:
        rets = np.log(weekly / weekly.shift(1))
    else:
        rets = weekly.pct_change()

    rets = rets.dropna()
    returns_tensor = torch.tensor(rets.values, dtype=torch.float32)
    return returns_tensor, rets.index, rets.to_frame(name="return")


def create_event_labels(returns_1d: torch.Tensor, threshold: float = -0.02) -> torch.Tensor:
    return (returns_1d < threshold).to(torch.float32)


# =========================
# CONV (multi-kernel, causal)
# =========================
class MultiScaleCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernels: Sequence[int],
        out_channels_per_kernel: int,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        assert len(kernels) >= 1
        assert all(k >= 1 for k in kernels)

        self.kernels: List[int] = list(kernels)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels_per_kernel,
                kernel_size=k,
                stride=1,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ) for k in self.kernels
        ])
        self.left_paddings = [(k - 1) * dilation for k in self.kernels]
        self.out_channels = out_channels_per_kernel * len(self.kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = []
        for conv, pad_left in zip(self.convs, self.left_paddings):
            x_pad = F.pad(x, (pad_left, 0))
            ys.append(conv(x_pad))
        return torch.cat(ys, dim=1)


# =========================
# SURROGATE SPIKE
# =========================
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma: float = 1.0):
        ctx.save_for_backward(x)
        ctx.gamma = gamma
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        gamma = ctx.gamma
        ax = (gamma * x).abs()
        grad = (ax < 1.0).to(x.dtype) * (1.0 - ax) * gamma
        return grad_output * grad, None


# =========================
# LIF BANK (NOW: refractory + adaptive threshold)
# =========================
class LIFBank(nn.Module):
    def __init__(
        self,
        num_neurons: int,
        alpha: float = 0.95,
        theta: float = 1.0,          # <<< raise threshold (we normalize input now)
        v_reset: float = 0.0,
        hard_reset: bool = False,
        surrogate_gamma: float = 8.0,
        learn_theta: bool = False,
        refractory_steps: int = 2,    # <<< NEW: prevents rapid re-spiking
        adaptive_theta: bool = True,  # <<< NEW: makes firing naturally sparse
        theta_inc: float = 0.35,      # <<< increase threshold after spike
        theta_decay: float = 0.995,   # <<< decay back to base
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.alpha = float(alpha)
        self.hard_reset = bool(hard_reset)
        self.v_reset = float(v_reset)
        self.surrogate_gamma = float(surrogate_gamma)

        self.refractory_steps = int(refractory_steps)
        self.adaptive_theta = bool(adaptive_theta)
        self.theta_inc = float(theta_inc)
        self.theta_decay = float(theta_decay)

        if learn_theta:
            self.theta_base = nn.Parameter(torch.full((1, self.num_neurons, 1), float(theta)))
        else:
            self.register_buffer("theta_base", torch.full((1, self.num_neurons, 1), float(theta)))

        # state
        self.register_buffer("_v", None, persistent=False)
        self.register_buffer("_theta_dyn", None, persistent=False)
        self.register_buffer("_ref", None, persistent=False)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        self._v = torch.zeros((batch_size, self.num_neurons, 1), device=device, dtype=dtype)
        self._theta_dyn = self.theta_base.detach().to(device=device, dtype=dtype).expand(batch_size, -1, -1).clone()
        self._ref = torch.zeros((batch_size, self.num_neurons, 1), device=device, dtype=dtype)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert u.dim() == 3 and u.size(1) == self.num_neurons
        B, _, T = u.shape

        if (
            self._v is None
            or self._v.size(0) != B
            or self._v.device != u.device
            or self._v.dtype != u.dtype
        ):
            self.reset_state(B, u.device, u.dtype)

        v = self._v
        theta_dyn = self._theta_dyn
        ref = self._ref

        spikes = []
        v_hist = []

        # ensure base theta on correct device/dtype
        theta_base = self.theta_base.to(device=u.device, dtype=u.dtype)

        for t in range(T):
            # refractory gate: if ref>0, block input integration
            ref_active = (ref > 0).to(u.dtype)
            u_eff = u[:, :, t:t+1] * (1.0 - ref_active)

            v = self.alpha * v + u_eff

            # choose threshold
            theta = theta_dyn if self.adaptive_theta else theta_base

            s = SurrogateSpike.apply(v - theta, self.surrogate_gamma)

            # reset
            if self.hard_reset:
                v = torch.where(s > 0, torch.full_like(v, self.v_reset), v)
            else:
                v = v - s * theta

            # update refractory counter
            if self.refractory_steps > 0:
                ref = torch.maximum(ref - 1.0, torch.zeros_like(ref))
                ref = torch.where(s > 0, torch.full_like(ref, float(self.refractory_steps)), ref)

            # adaptive threshold update
            if self.adaptive_theta:
                theta_dyn = theta_dyn * self.theta_decay + theta_base * (1.0 - self.theta_decay)
                theta_dyn = theta_dyn + self.theta_inc * s

            spikes.append(s)
            v_hist.append(v)

        spikes = torch.cat(spikes, dim=2)
        v_hist = torch.cat(v_hist, dim=2)

        self._v = v.detach()
        self._theta_dyn = theta_dyn.detach()
        self._ref = ref.detach()

        return spikes, v_hist


# =========================
# MODEL
# =========================
class ConvLIFEventModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernels: Sequence[int],
        out_channels_per_kernel: int,
        lif_alpha: float = 0.95,
        lif_theta: float = 1.0,
        lif_hard_reset: bool = False,
        lif_v_reset: float = 0.0,
        surrogate_gamma: float = 8.0,
        learn_theta: bool = False,
        refractory_steps: int = 2,
        adaptive_theta: bool = True,
        theta_inc: float = 0.35,
        theta_decay: float = 0.995,
    ):
        super().__init__()
        self.conv = MultiScaleCausalConv1d(
            in_channels=in_channels,
            kernels=kernels,
            out_channels_per_kernel=out_channels_per_kernel,
        )
        self.lif = LIFBank(
            num_neurons=self.conv.out_channels,
            alpha=lif_alpha,
            theta=lif_theta,
            hard_reset=lif_hard_reset,
            v_reset=lif_v_reset,
            surrogate_gamma=surrogate_gamma,
            learn_theta=learn_theta,
            refractory_steps=refractory_steps,
            adaptive_theta=adaptive_theta,
            theta_inc=theta_inc,
            theta_decay=theta_decay,
        )

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        self.lif.reset_state(batch_size, device, dtype)

    def forward(self, x: torch.Tensor):
        u = self.conv(x)
        spikes, v_hist = self.lif(u)
        return u, spikes, v_hist


# =========================
# LOSS (NEW: logits from v-theta, plus sparsity)
# =========================
class EventAlignmentSparsityLoss(nn.Module):
    """
    Align events using z_max(t) = max_f (v - theta) as a continuous logit,
    then enforce sparsity + penalize false positives outside ±H neighborhood.
    """
    def __init__(
        self,
        H: int = 8,
        lambda_fp: float = 1.0,     # <<< stronger FP penalty
        lambda_rate: float = 2e-2,  # <<< global sparsity
        lambda_balance: float = 1e-3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.H = int(H)
        self.lambda_fp = float(lambda_fp)
        self.lambda_rate = float(lambda_rate)
        self.lambda_balance = float(lambda_balance)
        self.eps = float(eps)

    def forward(self, spikes: torch.Tensor, v_hist: torch.Tensor, theta_base: torch.Tensor, y: torch.Tensor):
        """
        spikes: [B,F,T] (hard forward but surrogate gradients ok)
        v_hist: [B,F,T]
        theta_base: [1,F,1] or [B,F,1]
        y: [B,T] in {0,1}
        """
        B, F_, T = spikes.shape
        dtype = v_hist.dtype
        device = v_hist.device

        y = y.to(dtype)

        # build neighborhood mask N via 1D conv on y
        # y: [B,1,T]
        k = 2 * self.H + 1
        kernel = torch.ones(1, 1, k, device=device, dtype=dtype)
        N = F.conv1d(y[:, None, :], kernel, padding=self.H)  # [B,1,T]
        N = (N > 0).to(dtype)[:, 0, :]                       # [B,T] in {0,1}

        # continuous logit for "event-likeliness": z_max(t)=max_f(v - theta_base)
        # NOTE: we use theta_base (not theta_dyn) to keep target stable
        theta = theta_base.to(device=device, dtype=dtype)
        z = v_hist - theta  # [B,F,T]
        z_max = z.max(dim=1).values  # [B,T] continuous

        # normalize per-sample so BCE is well-behaved
        z_mu = z_max.mean(dim=1, keepdim=True)
        z_std = z_max.std(dim=1, keepdim=True) + 1e-6
        z_norm = (z_max - z_mu) / z_std

        # alignment: we want logits high in neighborhood, low outside
        loss_align = F.binary_cross_entropy_with_logits(z_norm, N)

        # false positives outside neighborhood: any spike at t counts
        s_any = (spikes.sum(dim=1) > 0).to(dtype)  # [B,T]
        outside = (1.0 - N)
        outside_sum = outside.sum()
        loss_fp = (s_any * outside).sum() / (outside_sum + self.eps) if outside_sum > 0 else torch.zeros((), device=device, dtype=dtype)

        # global sparsity: penalize raw spike rate
        loss_rate = spikes.to(dtype).mean()

        # neuron balance: prevent "one neuron always firing"
        per_neuron_rate = spikes.to(dtype).mean(dim=2)  # [B,F]
        loss_balance = per_neuron_rate.var(dim=1).mean()

        loss = loss_align + self.lambda_fp * loss_fp + self.lambda_rate * loss_rate + self.lambda_balance * loss_balance

        terms = {
            "loss_align": loss_align.detach(),
            "loss_fp": loss_fp.detach(),
            "loss_rate": loss_rate.detach(),
            "loss_balance": loss_balance.detach(),
            "spikes_any_rate": s_any.mean().detach(),
            "events": (y > 0.5).sum().detach(),
        }
        return loss, terms


# =========================
# DATASET (NEW: per-window normalization)
# =========================
class SlidingWindowDataset(Dataset):
    def __init__(self, returns_1d: torch.Tensor, labels_1d: torch.Tensor, W: int, stride: int = 1, normalize: bool = True):
        super().__init__()
        assert returns_1d.dim() == 1 and labels_1d.dim() == 1
        assert returns_1d.shape[0] == labels_1d.shape[0]
        self.r = returns_1d
        self.y = labels_1d
        self.W = int(W)
        self.stride = int(stride)
        self.normalize = bool(normalize)
        self.starts = list(range(0, len(self.r) - self.W + 1, self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.W
        x = self.r[s:e].unsqueeze(0)  # [1,W]
        y = self.y[s:e]              # [W]

        if self.normalize:
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True) + 1e-6
            x = (x - mu) / sd

        return x, y


# =========================
# TRAIN
# =========================
def train_conv_lif(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    grad_clip: float = 1.0,
    debug_first_batch: bool = True,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "test_loss": []}

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        total, n = 0.0, 0
        printed = False

        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float()

            if train:
                optimizer.zero_grad(set_to_none=True)

            model.reset_state(batch_size=x.size(0), device=x.device, dtype=x.dtype)
            _, spikes, v_hist = model(x)

            loss, terms = loss_fn(spikes=spikes, v_hist=v_hist, theta_base=model.lif.theta_base, y=y)

            if train:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

            if debug_first_batch and (not printed) and train:
                printed = True
                with torch.no_grad():
                    print(
                        f"[debug] spike_rate={spikes.mean().item():.6f} "
                        f"spikes_any_rate={terms['spikes_any_rate'].item():.6f} "
                        f"v_mean={v_hist.mean().item():.6f} v_max={v_hist.max().item():.6f} "
                        f"theta_base={float(model.lif.theta_base.mean().item()):.4f} "
                        f"events_in_batch={int((y>0.5).sum().item())} "
                        f"loss_align={terms['loss_align'].item():.4f} "
                        f"loss_fp={terms['loss_fp'].item():.4f} "
                        f"loss_rate={terms['loss_rate'].item():.4f}"
                    )

        return total / max(1, n)

    for ep in range(1, epochs + 1):
        tr = run_epoch(train_loader, train=True)
        te = run_epoch(test_loader, train=False)
        history["train_loss"].append(tr)
        history["test_loss"].append(te)
        print(f"Epoch {ep:03d} | train {tr:.5f} | test {te:.5f}")

    return history


# =========================
# PLOTS
# =========================
@torch.no_grad()
def plot_loss_history(history):
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["test_loss"], label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss history")
    plt.show()


@torch.no_grad()
def plot_spikes_vs_events(model, dataset, idx: int | None = None, device: str = "cpu", max_neurons: int = 12):
    model.eval()
    model.to(device)

    if idx is None:
        idx = len(dataset) // 2

    x, y = dataset[idx]
    x = x.unsqueeze(0).to(device)  # [1,1,W]
    y = y.unsqueeze(0).to(device)  # [1,W]

    model.reset_state(batch_size=1, device=x.device, dtype=x.dtype)
    _, spikes, _ = model(x)

    spikes = spikes[0].cpu()  # [F,W]
    y = y[0].cpu()            # [W]

    F_, W_ = spikes.shape
    K = min(F_, max_neurons)

    plt.figure()
    event_idx = torch.where(y > 0.5)[0].numpy()
    if len(event_idx) > 0:
        plt.scatter(event_idx, [K + 1] * len(event_idx), marker="x", label="TARGET (return<-2%)")

    for f in range(K):
        t_idx = torch.where(spikes[f] > 0.5)[0].numpy()
        if len(t_idx) > 0:
            plt.scatter(t_idx, [f] * len(t_idx), marker="|")

    plt.yticks(list(range(K)) + [K + 1], [f"lif {i}" for i in range(K)] + ["TARGET"])
    plt.xlabel("t within window")
    plt.title("Hard spikes vs target events (one window)")
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    returns_raw, dates, _ = load_sp500_weekly_returns(start="1950-01-01", use_log_returns=False)
    labels = create_event_labels(returns_raw, threshold=-0.02)

    # IMPORTANT CHANGE:
    # No more returns * 50.0 — we normalize per window in the dataset.
    returns = returns_raw.clone()

    T_total = len(returns)
    split = int(0.8 * T_total)
    r_train, r_test = returns[:split], returns[split:]
    y_train, y_test = labels[:split], labels[split:]

    W = 256
    train_ds = SlidingWindowDataset(r_train, y_train, W=W, stride=1, normalize=True)
    test_ds = SlidingWindowDataset(r_test, y_test, W=W, stride=1, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)

    model = ConvLIFEventModel(
        in_channels=1,
        kernels=[8, 16, 32],
        out_channels_per_kernel=8,
        lif_alpha=0.95,
        lif_theta=1.0,          # <<< raised due to normalization
        lif_hard_reset=False,
        surrogate_gamma=8.0,
        learn_theta=False,
        refractory_steps=2,
        adaptive_theta=True,
        theta_inc=0.35,
        theta_decay=0.995,
    )

    loss_fn = EventAlignmentSparsityLoss(
        H=8,
        lambda_fp=1.0,
        lambda_rate=2e-2,
        lambda_balance=1e-3,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    history = train_conv_lif(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
        lr=1e-3,
        device=device,
        grad_clip=1.0,
        debug_first_batch=True,
    )

    plot_loss_history(history)
    plot_spikes_vs_events(model, test_ds, idx=len(test_ds) // 2, device=device, max_neurons=12)