import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Sequence, List, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################ CONV LAYER ################
class MultiScaleCausalConv1d(nn.Module):
    """
    Multi-Scale kausale Conv1d Filterbank (parallel kernels).
    Jeder Branch hat eigene kernel_size, aber gleiche out_channels_per_kernel.
    Outputs werden über Channel-Dimension konkateniert.

    Input:  x [B, C_in, T]
    Output: y [B, F_total, T]  mit F_total = len(kernels) * out_channels_per_kernel

    Kausalität: Output[t] hängt nur von Input[<=t] ab (left-padding).
    """
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
        assert len(kernels) >= 1, "kernels must be non-empty"
        assert all(k >= 1 for k in kernels), "all kernel sizes must be >= 1"
        assert out_channels_per_kernel >= 1
        assert dilation >= 1

        self.kernels: List[int] = list(kernels)
        self.dilation = dilation

        # pro Kernelgröße ein Conv-Branch
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels_per_kernel,
                kernel_size=k,
                stride=1,
                padding=0,      # wir padden manuell kausal
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            for k in self.kernels
        ])

        # precompute left paddings (pro Kernel unterschiedlich)
        self.left_paddings = [(k - 1) * dilation for k in self.kernels]

        self.out_channels = out_channels_per_kernel * len(self.kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        ys = []
        for conv, pad_left in zip(self.convs, self.left_paddings):
            x_pad = F.pad(x, (pad_left, 0))   # (pad_left, pad_right)
            ys.append(conv(x_pad))            # [B, F_k, T]
        return torch.cat(ys, dim=1)           # [B, F_total, T]


################ LIF LAYER ################
class SurrogateSpike(torch.autograd.Function):
    """
    Hard threshold in forward:
        s = 1 if x >= 0 else 0
    Surrogate gradient in backward:
        ds/dx ≈ triangular window around 0
    """
    @staticmethod
    def forward(ctx, x, gamma: float = 1.0):
        # x: any shape
        ctx.save_for_backward(x)
        ctx.gamma = gamma
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        gamma = ctx.gamma

        # piecewise-linear surrogate: nonzero in |x| < 1/gamma
        # slope height scales with gamma
        ax = (gamma * x).abs()
        grad = (ax < 1.0).to(x.dtype) * (1.0 - ax) * gamma  # triangle
        return grad_output * grad, None

    import torch
    import torch.nn as nn
    from typing import Optional, Tuple

    class LIFBank(nn.Module):
        """
        Vectorisierte LIF-Neuron-Bank (1:1 zu Conv-Channels).
        Input u: [B, F, T]
        Output spikes: [B, F, T], v: [B, F, T]

        Forward: hard spikes
        Backward: surrogate via SurrogateSpike
        """

        def __init__(
                self,
                num_neurons: int,  # F
                alpha: float = 0.95,  # leak (exp(-dt/tau))
                theta: float = 1.0,  # threshold (can be scalar or learnable later)
                v_reset: float = 0.0,  # optional hard reset value
                hard_reset: bool = False,  # if True: v <- v_reset after spike; else subtract theta
                surrogate_gamma: float = 5.0,
                learn_theta: bool = False,
        ):
            super().__init__()
            self.F = num_neurons
            self.alpha = float(alpha)
            self.hard_reset = bool(hard_reset)
            self.v_reset = float(v_reset)
            self.surrogate_gamma = float(surrogate_gamma)

            if learn_theta:
                self.theta = nn.Parameter(torch.full((1, self.F, 1), float(theta)))
            else:
                self.register_buffer("theta", torch.full((1, self.F, 1), float(theta)))

            self.register_buffer("_v", None, persistent=False)  # state

        def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
            self._v = torch.zeros((batch_size, self.F, 1), device=device, dtype=dtype)

        def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            u: [B, F, T]
            returns:
              spikes: [B, F, T]
              v_hist: [B, F, T]
            """
            assert u.dim() == 3 and u.size(1) == self.F, "u must be [B, F, T]"
            B, F, T = u.shape
            device, dtype = u.device, u.dtype

            # init/reset state if needed or batch size changed
            if self._v is None or self._v.size(0) != B or self._v.device != device or self._v.dtype != dtype:
                self.reset_state(B, device, dtype)

            v = self._v  # [B, F, 1]
            spikes = []
            v_hist = []

            for t in range(T):
                # leaky integrate
                v = self.alpha * v + u[:, :, t:t + 1]

                # hard spike forward, surrogate backward
                s = SurrogateSpike.apply(v - self.theta, self.surrogate_gamma)  # [B,F,1]

                # reset
                if self.hard_reset:
                    v = torch.where(s > 0, torch.full_like(v, self.v_reset), v)
                else:
                    v = v - s * self.theta  # subtractive reset

                spikes.append(s)
                v_hist.append(v)

            spikes = torch.cat(spikes, dim=2)  # [B, F, T]
            v_hist = torch.cat(v_hist, dim=2)  # [B, F, T]

            # store last state (for streaming / TBPTT)
            self._v = v.detach()  # detach by default to avoid exploding graph across batches

            return spikes, v_hist

        def get_spike_prob(self, v_hist: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
            """
            Optional helper for loss: converts membrane potential to soft spike probability.
            p = sigmoid(beta*(v-theta))
            v_hist: [B, F, T]
            returns p: [B, F, T]
            """
            return torch.sigmoid(beta * (v_hist - self.theta))



############ DATA #############


def load_sp500_weekly_returns(
        start: str = "1950-01-01",
        end: str = None,
        use_log_returns: bool = True,
):
    """
    Lädt S&P 500 (^GSPC) von Yahoo Finance
    Resampled auf Weekly (Freitag)
    Berechnet Returns

    Returns:
        returns_tensor: torch.Tensor [T]
        dates: pd.DatetimeIndex
        raw_df: pandas DataFrame mit Weekly Preisen + Returns
    """

    ticker = "^GSPC"

    # --- Download daily data ---
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError("No data downloaded from Yahoo.")

    # --- Resample to weekly (Friday close) ---
    weekly = df["Adj Close"].resample("W-FRI").last().dropna()

    # --- Compute returns ---
    if use_log_returns:
        returns = np.log(weekly / weekly.shift(1))
    else:
        returns = weekly.pct_change()

    returns = returns.dropna()

    # --- Convert to torch tensor ---
    returns_tensor = torch.tensor(returns.values, dtype=torch.float32)

    return returns_tensor, returns.index, returns.to_frame(name="return")



############## LOSS ###########

class NearestHardSpikeLoss(nn.Module):
    """
    - Assignment basiert NUR auf hard spikes s (0/1).
    - Für jedes Event wird der zeitlich nächstgelegene Spike im ±H Fenster gewählt
      (Neuron f*, Zeit t*).
    - Gradienten gehen NUR über v_hist[b,f*,t*] (surrogate wirkt dort automatisch,
      weil v aus dem LIF-Graph kommt).
    - Wenn es KEINEN Spike im Fenster gibt: nimm das Maximum von (v-theta) im Fenster
      und drücke es nach oben (damit ein Spike entsteht).
    - Zusätzlich: FP penalty außerhalb Event-Nachbarschaften, plus Homeostasis.
    """
    def __init__(
        self,
        H: int = 8,
        margin: float = 0.2,        # wie "weit über threshold" wir den Spike pushen wollen
        lambda_miss: float = 1.0,   # wenn kein Spike im Fenster existiert
        lambda_fp: float = 0.2,     # spikes außerhalb Event-Nachbarschaften
        lambda_homeo: float = 0.05, # gleichmäßige Nutzung der Neuronen (in Event-Nbhds)
        eps: float = 1e-6,
    ):
        super().__init__()
        self.H = int(H)
        self.margin = float(margin)
        self.lambda_miss = float(lambda_miss)
        self.lambda_fp = float(lambda_fp)
        self.lambda_homeo = float(lambda_homeo)
        self.eps = float(eps)

    def forward(self, spikes: torch.Tensor, v_hist: torch.Tensor, theta: torch.Tensor, y: torch.Tensor):
        """
        spikes: [B,F,T] hard 0/1 (aus forward)
        v_hist: [B,F,T] membrane
        theta:  [1,F,1]
        y:      [B,T] 0/1 events
        """
        B, F, T = spikes.shape
        device = spikes.device
        dtype = v_hist.dtype

        # ---------- Neighborhood mask N (±H um jedes Event) ----------
        N = torch.zeros((B, T), device=device, dtype=dtype)
        if (y > 0.5).any():
            for dt in range(-self.H, self.H + 1):
                if dt < 0:
                    N[:, :dt] = torch.maximum(N[:, :dt], y[:, -dt:])
                elif dt > 0:
                    N[:, dt:] = torch.maximum(N[:, dt:], y[:, :-dt])
                else:
                    N = torch.maximum(N, y)

        # ---------- 1) Event->nearest spike assignment ----------
        event_push_losses = []
        miss_losses = []

        # We work with z = v - theta (same shape [B,F,T])
        z = v_hist - theta  # broadcast

        for b in range(B):
            ev_idx = torch.where(y[b] > 0.5)[0]
            if ev_idx.numel() == 0:
                continue

            for e in ev_idx.tolist():
                lo = max(0, e - self.H)
                hi = min(T - 1, e + self.H)

                # candidate spikes in window: indices where any neuron spikes
                # mask shape [F, L]
                s_win = spikes[b, :, lo:hi+1]  # [F,L]
                if s_win.sum().item() > 0:
                    # find nearest in time (min |t-e|), tie-break: smaller |dt| then earlier time then first neuron
                    L = hi - lo + 1
                    t_grid = torch.arange(lo, hi+1, device=device)  # [L]
                    dist = (t_grid - e).abs()  # [L]

                    # For each time position, check if any neuron spiked there
                    any_spike_t = (s_win.sum(dim=0) > 0)  # [L] bool
                    # take minimal distance among times where any_spike_t True
                    dist_masked = dist.clone()
                    dist_masked[~any_spike_t] = 10**9
                    t_star = t_grid[torch.argmin(dist_masked)].item()

                    # choose a neuron that spiked at t_star (first one)
                    f_candidates = torch.where(spikes[b, :, t_star] > 0.5)[0]
                    f_star = f_candidates[0].item()

                    # push z[b,f*,t*] to be >= margin (i.e., comfortably above threshold)
                    # hinge: max(0, margin - z)
                    event_push_losses.append(F.relu(self.margin - z[b, f_star, t_star]))
                else:
                    # no spike in window -> MISS:
                    # take the maximal z in window and push it above margin
                    z_win = z[b, :, lo:hi+1]  # [F,L]
                    z_max = z_win.max()
                    miss_losses.append(F.relu(self.margin - z_max))

        if len(event_push_losses) == 0:
            L_event = torch.zeros((), device=device, dtype=dtype)
        else:
            L_event = torch.stack(event_push_losses).mean()

        if len(miss_losses) == 0:
            L_miss = torch.zeros((), device=device, dtype=dtype)
        else:
            L_miss = torch.stack(miss_losses).mean()

        # ---------- 2) False positives outside neighborhoods ----------
        outside = (1.0 - N)  # [B,T]
        denom_out = outside.sum() + self.eps

        # any spike at time t?
        s_any = (spikes.sum(dim=1) > 0).to(dtype)  # [B,T]
        L_fp = (s_any * outside).sum() / denom_out

        # ---------- 3) Homeostasis inside neighborhoods ----------
        # Use spike counts (hard) inside N to equalize neuron usage
        inside = N  # [B,T]
        denom_in = inside.sum() + self.eps

        # activity per neuron = mean spike probability inside neighborhood
        # since spikes are hard 0/1: use spike rate
        a_f = (spikes.to(dtype) * inside[:, None, :]).sum(dim=(0, 2)) / denom_in  # [F]
        L_homeo = ((a_f - a_f.mean()) ** 2).mean()

        loss = L_event + self.lambda_miss * L_miss + self.lambda_fp * L_fp + self.lambda_homeo * L_homeo

        terms = {
            "L_event": L_event.detach(),
            "L_miss": L_miss.detach(),
            "L_fp": L_fp.detach(),
            "L_homeo": L_homeo.detach(),
            "events": (y > 0.5).sum().detach(),
            "spikes_any_rate": s_any.mean().detach(),
        }
        return loss, terms


####### TRAINING #########
class SlidingWindowDataset(Dataset):
    """
    returns: [T_total]
    labels:  [T_total] (0/1)
    liefert windows:
      x: [1, W]
      y: [W]
    """
    def __init__(self, returns_1d: torch.Tensor, labels_1d: torch.Tensor, W: int, stride: int = 1):
        super().__init__()
        assert returns_1d.dim() == 1
        assert labels_1d.dim() == 1
        assert returns_1d.shape[0] == labels_1d.shape[0]
        self.r = returns_1d
        self.y = labels_1d
        self.W = int(W)
        self.stride = int(stride)

        self.starts = list(range(0, len(self.r) - self.W + 1, self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.W
        x = self.r[s:e].unsqueeze(0)      # [1, W]
        y = self.y[s:e]                  # [W]
        return x, y



def train_conv_lif(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    grad_clip: float = 1.0,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_loss": [],
    }

    def run_epoch(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        total = 0.0
        n = 0

        for x, y in loader:
            # x: [B,1,W], y: [B,W]
            x = x.to(device)
            y = y.to(device).float()

            if train:
                optimizer.zero_grad(set_to_none=True)

            # Für Sliding-Window-Training: state pro Batch resetten (stabiler Baseline)
            model.reset_state(batch_size=x.size(0), device=x.device, dtype=x.dtype)

            # forward
            u, spikes, v_hist = model(x)

            # loss: nutzt hard spikes + v_hist
            loss, terms = loss_fn(spikes=spikes, v_hist=v_hist, theta=model.lif.theta, y=y)

            if train:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

        return total / max(1, n)

    for ep in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        test_loss = run_epoch(test_loader, train=False)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        print(f"Epoch {ep:03d} | train {train_loss:.5f} | test {test_loss:.5f}")

    return history


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
def plot_spikes_vs_events(model, dataset, idx: int = None, device: str = "cpu", max_neurons: int = 12):
    """
    Plottet für ein einzelnes Window:
      - Target events (y[t]=1)
      - Spike trains für einige LIF-Neuronen
    """
    model.eval()
    model.to(device)

    if idx is None:
        idx = len(dataset) // 2

    x, y = dataset[idx]
    # make batch size 1
    x = x.unsqueeze(0).to(device)      # [1,1,W]
    y = y.unsqueeze(0).to(device)      # [1,W]

    model.reset_state(batch_size=1, device=x.device, dtype=x.dtype)
    u, spikes, v_hist = model(x)

    # spikes: [1,F,W]
    spikes = spikes[0].detach().cpu()  # [F,W]
    y = y[0].detach().cpu()            # [W]

    F, W = spikes.shape
    K = min(F, max_neurons)

    plt.figure()
    # plot target events as stem-like markers
    event_idx = torch.where(y > 0.5)[0].numpy()
    if len(event_idx) > 0:
        plt.scatter(event_idx, [K + 1] * len(event_idx), marker="x", label="target event (return<-2%)")

    # plot spikes for first K neurons as raster
    for f in range(K):
        t_idx = torch.where(spikes[f] > 0.5)[0].numpy()
        if len(t_idx) > 0:
            plt.scatter(t_idx, [f] * len(t_idx), marker="|")

    plt.yticks(list(range(K)) + [K + 1], [f"lif {i}" for i in range(K)] + ["TARGET"])
    plt.xlabel("t within window")
    plt.title("Spikes (hard) vs Target Events (one window)")
    plt.show()

    # =========================
    # 3) TRAINING BLOCK
    # =========================

    import torch
    from torch.utils.data import DataLoader

    # ---- Chronologischer Split ----
    T = len(returns)
    split = int(0.8 * T)

    r_train = returns[:split]
    r_test = returns[split:]

    y_train = labels[:split]
    y_test = labels[split:]

    # ---- Sliding Windows ----
    W = 256  # Window length
    train_ds = SlidingWindowDataset(r_train, y_train, W=W, stride=1)
    test_ds = SlidingWindowDataset(r_test, y_test, W=W, stride=1)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)

    # ---- Model ----
    model = ConvLIFEventModel(
        in_channels=1,
        kernels=[8, 16, 32],
        out_channels_per_kernel=8,  # 3*8 = 24 LIF neurons
        lif_alpha=0.95,
        lif_theta=1.0,
        lif_hard_reset=False,
        surrogate_gamma=5.0,
        learn_theta=False,
    )

    # ---- Loss ----
    loss_fn = NearestHardSpikeLoss(
        H=8,
        margin=0.2,
        lambda_miss=1.0,
        lambda_fp=0.2,
        lambda_homeo=0.05,
    )

    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- Training ----
    history = train_conv_lif(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
        lr=1e-3,
        device=device,
        grad_clip=1.0,
    )

    print("Training finished.")


# =========================
# PLOTS
# =========================

plot_loss_history(history)

plot_spikes_vs_events(
    model=model,
    dataset=test_ds,
    idx=len(test_ds)//2,
    device=device,
    max_neurons=12
)