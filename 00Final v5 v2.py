import yfinance as yf, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import math

# =========================
# data
# =========================
def make_xy(r, L, event_thr=-0.02):
    X = r.unfold(0, L, 1).unsqueeze(1)           # [N,1,L]
    Y = (r < event_thr).float().unfold(0, L, 1)  # [N,L]
    return X, Y

def sp500_loaders(start="2000-01-01", L=52, B=64, split=0.8, event_thr=-0.02):
    close = yf.download("^GSPC", start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(close, pd.DataFrame): close = close.squeeze()
    r = torch.tensor(close.resample("W-FRI").last().pct_change().dropna().values, dtype=torch.float32)
    cut = int(split * len(r))
    Xtr, Ytr = make_xy(r[:cut], L, event_thr)
    Xte, Yte = make_xy(r[cut:], L, event_thr)
    return (
        DataLoader(TensorDataset(Xtr, Ytr), batch_size=B, shuffle=True),
        DataLoader(TensorDataset(Xte, Yte), batch_size=B, shuffle=False),
        r
    )

# =========================
# model
# =========================
class CausalConvBank(nn.Module):
    def __init__(self, n_filters, kernel_size):
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv1d(1, n_filters, kernel_size, bias=False)

    def forward(self, x):
        w = self.conv.weight
        w = w / w.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8).view(-1, 1, 1)  # per-kernel L2 = 1
        return F.conv1d(F.pad(x, (self.k - 1, 0)), w)  # [B,F,L]

class LIFLayer(nn.Module):
    def __init__(self, n_filters, tau=3.0, threshold=0.25, beta=15.0, dt=1.0):
        super().__init__()
        self.dt = float(dt)
        self.register_buffer("thr", torch.tensor(float(threshold), dtype=torch.float32))
        self.beta = float(beta)
        raw0 = math.log(math.expm1(float(tau)))          # inverse softplus so tau starts at given value
        self.raw_tau = nn.Parameter(torch.full((n_filters,), raw0, dtype=torch.float32))

    def tau(self):
        return F.softplus(self.raw_tau) + 1e-4           # [F] > 0

    def forward(self, I):                                # I: [B,F,L]
        B, F_, L = I.shape
        v = I.new_zeros(B, F_)
        z_hist, s_hist = [], []

        alpha = torch.exp(-self.dt / self.tau()).to(I.dtype).view(1, F_)   # [1,F]
        thr, beta = self.thr.to(I.dtype), self.beta

        for t in range(L):
            v_pre = alpha * v + (1 - alpha) * I[:, :, t]
            z = beta * (v_pre - thr)
            s = (v_pre >= thr).to(I.dtype)
            v = v_pre * (1 - s)
            z_hist.append(z)
            s_hist.append(s)

        return torch.stack(z_hist, -1), torch.stack(s_hist, -1)

class SNNEventModel(nn.Module):
    def __init__(
        self,
        n_filters=8,
        kernel_size=8,
        tau=3.0,
        threshold=0.25,
        beta=15.0,
        x_scale=20.0,
        kp=10.0,
        gain_min=0.25,
        gain_max=4.0,
    ):
        super().__init__()
        self.x_scale = x_scale
        self.bank = CausalConvBank(n_filters, kernel_size)
        self.lif = LIFLayer(n_filters, tau=tau, threshold=threshold, beta=beta)

        self.kp = float(kp)
        self.gain_min, self.gain_max = float(gain_min), float(gain_max)
        self.register_buffer("gain", torch.tensor(1.0))

    def forward(self, x):
        I = self.gain * self.bank(self.x_scale * x)
        z, s = self.lif(I)
        return {"I": I, "z": z, "s": s, "logits": z.amax(dim=1)}  # [B,L]

    @torch.no_grad()
    def p_update_epoch(self, spike_sum, target_sum, spike_sites):
        err = target_sum / spike_sites - spike_sum / spike_sites
        self.gain.add_(self.kp * err).clamp_(self.gain_min, self.gain_max)
        return err

# =========================
# loss
# =========================
def event_loss(logits, z, y, lambda_neg=0.1, lambda_cdf=0.5, lambda_dup=0.05):
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)

    # local event classification
    pos_term = -(y * F.logsigmoid(logits)).sum() / pos
    neg_term = -((1.0 - y) * F.logsigmoid(-logits)).sum() / neg
    L_event = pos_term + lambda_neg * neg_term

    # cumulative alignment
    p = torch.sigmoid(logits)
    L_cdf = (p.cumsum(dim=1) - y.cumsum(dim=1)).abs().mean()

    # duplicate penalty across filters at same time
    u = torch.sigmoid(z).sum(dim=1)             # [B,L]
    L_dup = F.relu(u - 1.0).pow(2).mean()

    return L_event + lambda_cdf * L_cdf + lambda_dup * L_dup

# =========================
# train / eval
# =========================
@torch.no_grad()
def evaluate(model, loader, device, lambda_neg=0.1, lambda_cdf=0.5, lambda_dup=0.05):
    model.eval()
    total, n, spk, tgt = 0.0, 0, 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total += event_loss(out["logits"], out["z"], yb, lambda_neg, lambda_cdf, lambda_dup).item() * xb.size(0)
        n += xb.size(0)
        spk += out["s"].sum().item()
        tgt += yb.sum().item()
    return total / n, spk, tgt

@torch.no_grad()
def confusion_metrics(model, loader, device="cpu"):
    """
    Event-level confusion metrics on the test set.
    A predicted event occurs whenever at least one LIF unit spikes at a time step.
    """
    model.eval()
    tp = fp = tn = fn = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)

        # event prediction: at least one spike across filters at a time step
        y_hat = (out["s"].sum(dim=1) > 0).to(torch.int64)   # [B,L]
        y_true = (yb > 0).to(torch.int64)                   # [B,L]

        tp += int(((y_hat == 1) & (y_true == 1)).sum().item())
        fp += int(((y_hat == 1) & (y_true == 0)).sum().item())
        tn += int(((y_hat == 0) & (y_true == 0)).sum().item())
        fn += int(((y_hat == 0) & (y_true == 1)).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

def fit(model, train_loader, test_loader, epochs=20, lr=1e-3, device="cpu",
        lambda_neg=0.1, lambda_cdf=0.5, lambda_dup=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    hist = {"train": [], "test": [], "gain": []}

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        train_spk, train_tgt, train_sites = 0.0, 0.0, 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = event_loss(out["logits"], out["z"], yb, lambda_neg, lambda_cdf, lambda_dup)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)
            n += xb.size(0)

            train_spk += out["s"].sum().item()
            train_tgt += yb.sum().item()
            train_sites += out["s"].numel()

        err = model.p_update_epoch(train_spk, train_tgt, train_sites)

        tr_loss = total / n
        te_loss, te_spk, te_tgt = evaluate(model, test_loader, device, lambda_neg, lambda_cdf, lambda_dup)

        hist["train"].append(tr_loss)
        hist["test"].append(te_loss)
        hist["gain"].append(model.gain.item())

        print(
            f"Epoch {ep:03d} | train {tr_loss:.4f} | test {te_loss:.4f} | "
            f"gain {model.gain.item():.3f} | train spikes {int(train_spk)} | train targets {int(train_tgt)} | "
            f"test spikes {int(te_spk)} | test targets {int(te_tgt)} | err {err:.5f}"
        )

    return model, hist

# =========================
# plots / diagnostics
# =========================
def plot_history(hist):
    plt.figure(figsize=(8,4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["test"], label="test")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss history")
    plt.legend(); plt.tight_layout(); plt.show()

@torch.no_grad()
def plot_sequence(model, loader, device="cpu", idx=0):
    model.eval()
    X, Y = loader.dataset.tensors
    x = X[idx:idx+1].to(device)
    y = Y[idx].cpu()
    out = model(x)
    s = out["s"][0].cpu()

    M = torch.cat([y.unsqueeze(0), s], dim=0).numpy()
    labels = ["target"] + [f"LIF {i}" for i in range(s.size(0))]

    plt.figure(figsize=(12, 4.5))
    plt.imshow(M, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("time step in sequence")
    plt.title(f"idx={idx} | gain={model.gain.item():.3f} | targets={int(y.sum())} | spikes={int(s.sum())}")
    plt.tight_layout()
    plt.show()

    print(f"idx {idx:03d} | targets {int(y.sum().item())} | spikes {int(s.sum().item())}")
    print("taus:", [round(t, 3) for t in model.lif.tau().detach().cpu().tolist()])
    print("Target positions:", torch.where(y > 0)[0].tolist())
    for i in range(s.size(0)):
        print(f"LIF {i} spikes at:", torch.where(s[i] > 0)[0].tolist())

@torch.no_grad()
def sample_spike_stats(model, loader, device="cpu", n=20):
    model.eval()
    s_counts, y_counts = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        s_counts.append(out["s"].sum(dim=(1, 2)).cpu())
        y_counts.append(yb.sum(dim=1).cpu())
    s_counts, y_counts = torch.cat(s_counts), torch.cat(y_counts)

    print(f"Mean spikes/sample:  {s_counts.float().mean().item():.2f}")
    print(f"Mean targets/sample: {y_counts.float().mean().item():.2f}")
    print(f"Zero-spike samples:  {int((s_counts == 0).sum())}/{len(s_counts)}")
    print(f"Zero-target samples: {int((y_counts == 0).sum())}/{len(y_counts)}")
    print("--- first samples ---")
    for i in range(min(n, len(s_counts))):
        print(f"idx {i:03d} | targets {int(y_counts[i].item())} | spikes {int(s_counts[i].item())}")
    return s_counts, y_counts

def top_target_indices(loader, n=6):
    Y = loader.dataset.tensors[1]
    return torch.topk(Y.sum(dim=1), k=min(n, len(Y))).indices.tolist()

@torch.no_grad()
def plot_sequences(model, loader, device="cpu", idxs=(0,1,2,3)):
    for idx in idxs:
        plot_sequence(model, loader, device=device, idx=int(idx))

# =========================
# run
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, returns = sp500_loaders(L=52, B=64, split=0.8, event_thr=-0.02)
model = SNNEventModel(
    n_filters=8,
    kernel_size=8,
    tau=3.0,
    threshold=0.25,
    beta=15.0,
    x_scale=20.0,
    kp=10.0,
    gain_min=0.25,
    gain_max=4.0,
)
model, hist = fit(
    model, train_loader, test_loader,
    epochs=100, lr=1e-3, device=device,
    lambda_neg=0.1, lambda_cdf=0.5, lambda_dup=0.05
)

plot_history(hist)
print("Learned taus:", [round(t, 3) for t in model.lif.tau().detach().cpu().tolist()])
sample_spike_stats(model, test_loader, device=device, n=12)
metrics = confusion_metrics(model, test_loader, device=device)
print(
    "Confusion counts | "
    f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}"
)
print(
    "Metrics | "
    f"Precision={metrics['precision']:.4f} "
    f"Recall={metrics['recall']:.4f} "
    f"F1={metrics['f1']:.4f} "
    f"Accuracy={metrics['accuracy']:.4f}"
)
print("Confusion matrix (rows=true, cols=pred):")
print(f"[[TN={metrics['tn']}, FP={metrics['fp']}],")
print(f" [FN={metrics['fn']}, TP={metrics['tp']}]]")
plot_sequences(model, test_loader, device=device, idxs=range(4))
plot_sequences(model, test_loader, device=device, idxs=top_target_indices(test_loader, n=4))
