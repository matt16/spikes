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
        self.conv = nn.Conv1d(1, n_filters, kernel_size, bias=True)

    def forward(self, x):
        return self.conv(F.pad(x, (self.k - 1, 0)))   # [B,F,L]

class LIFLayer(nn.Module):
    def __init__(self, tau=3.0, threshold=0.25, beta=10.0, dt=1.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(math.exp(-dt / tau), dtype=torch.float32))
        self.register_buffer("thr", torch.tensor(float(threshold), dtype=torch.float32))
        self.beta = float(beta)

    def forward(self, I):                       # I: [B,F,L]
        B, F_, L = I.shape
        v = I.new_zeros(B, F_)
        z_hist, s_hist = [], []

        alpha, thr, beta = self.alpha.to(I.dtype), self.thr.to(I.dtype), self.beta
        for t in range(L):
            v_pre = alpha * v + (1 - alpha) * I[:, :, t]   # exakte diskrete Lösung der DGL
            z = beta * (v_pre - thr)                       # Spike-Logit
            s = (v_pre >= thr).to(I.dtype)                # hard spike
            v = v_pre * (1 - s)                           # hard reset auf 0
            z_hist.append(z)
            s_hist.append(s)

        return torch.stack(z_hist, -1), torch.stack(s_hist, -1)

class SNNEventModel(nn.Module):
    def __init__(self, n_filters=8, kernel_size=8, tau=3.0, threshold=0.25, beta=10.0, x_scale=20.0):
        super().__init__()
        self.x_scale = x_scale
        self.bank = CausalConvBank(n_filters, kernel_size)
        self.lif = LIFLayer(tau=tau, threshold=threshold, beta=beta)

    def forward(self, x):
        I = self.bank(self.x_scale * x)         # feste Input-Skalierung
        z, s = self.lif(I)
        logits = z.amax(dim=1)                  # [B,L]
        return {"I": I, "z": z, "s": s, "logits": logits}

# =========================
# train / eval
# =========================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n, spk, tgt = 0.0, 0, 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total += criterion(out["logits"], yb).item() * xb.size(0)
        n += xb.size(0)
        spk += out["s"].sum().item()
        tgt += yb.sum().item()
    return total / n, spk, tgt

def fit(model, train_loader, test_loader, epochs=20, lr=1e-3, device="cpu"):
    ytr = train_loader.dataset.tensors[1]
    pos = ytr.sum().clamp_min(1.0)
    neg = ytr.numel() - pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=(neg / pos).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    hist = {"train": [], "test": []}

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb)["logits"], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)

        tr_loss = total / n
        te_loss, te_spk, te_tgt = evaluate(model, test_loader, criterion, device)
        hist["train"].append(tr_loss)
        hist["test"].append(te_loss)

        print(f"Epoch {ep:03d} | train {tr_loss:.4f} | test {te_loss:.4f} | test spikes {int(te_spk)} | test targets {int(te_tgt)}")

    return model, hist

# =========================
# plots
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
    s = out["s"][0].cpu()                      # [F,L]

    M = torch.cat([y.unsqueeze(0), s], dim=0).numpy()
    labels = ["target"] + [f"LIF {i}" for i in range(s.size(0))]

    plt.figure(figsize=(12, 4.5))
    plt.imshow(M, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("time step in sequence")
    plt.title("Targets and LIF spikes")
    plt.tight_layout()
    plt.show()

    print("Total target events:", int(y.sum().item()))
    print("Total spikes:", int(s.sum().item()))
    print("Target positions:", torch.where(y > 0)[0].tolist())
    for i in range(s.size(0)):
        print(f"LIF {i} spikes at:", torch.where(s[i] > 0)[0].tolist())

# =========================
# run
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, returns = sp500_loaders(L=52, B=64, split=0.8, event_thr=-0.02)
model = SNNEventModel(n_filters=8, kernel_size=8, tau=3.0, threshold=0.25, beta=10.0, x_scale=20.0)
model, hist = fit(model, train_loader, test_loader, epochs=100, lr=1e-3, device=device)

plot_history(hist)
plot_sequence(model, test_loader, device=device, idx=0)