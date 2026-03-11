import yfinance as yf, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# =========================
# data
# =========================
def make_xy(r, L, event_thr=-0.02):
    X = r.unfold(0, L, 1).unsqueeze(1)           # [N,1,L]
    Y = (r < event_thr).float().unfold(0, L, 1)  # [N,L]
    return X, Y

def sp500_loaders(start="2000-01-01", L=52, B=64, split=0.8, event_thr=-0.02):
    close = yf.download("^GSPC", start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    r = torch.tensor(close.resample("W-FRI").last().pct_change().dropna().values, dtype=torch.float32)

    cut = int(split * len(r))
    Xtr, Ytr = make_xy(r[:cut], L, event_thr)
    Xte, Yte = make_xy(r[cut:], L, event_thr)

    tr = DataLoader(TensorDataset(Xtr, Ytr), batch_size=B, shuffle=True)
    te = DataLoader(TensorDataset(Xte, Yte), batch_size=B, shuffle=False)
    return tr, te, r

# =========================
# model
# =========================
class CausalConvBank(nn.Module):
    def __init__(self, n_filters, kernel_size):
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv1d(1, n_filters, kernel_size, bias=True)

    def forward(self, x):                        # x: [B,1,L]
        return self.conv(F.pad(x, (self.k - 1, 0)))   # [B,F,L]

class LIFLayer(nn.Module):
    def __init__(self, tau=5.0, threshold=1.0, dt=1.0):
        super().__init__()
        self.register_buffer("k", torch.tensor(dt / tau, dtype=torch.float32))  # Euler: dt/tau
        self.register_buffer("threshold", torch.tensor(float(threshold), dtype=torch.float32))

    def forward(self, I):                        # I: [B,F,L]
        B, F_, L = I.shape
        v = I.new_zeros(B, F_)
        k = self.k.to(I.dtype)
        thr = self.threshold.to(I.dtype)

        v_pre_hist, v_hist, s_hist = [], [], []
        for t in range(L):                       # rekursiv -> Schleife über Zeit nötig
            v_pre = v + k * (-v + I[:, :, t])   # dV/dt = (-V + I)/tau
            s = (v_pre >= thr).to(I.dtype)      # harte Spikes
            v = v_pre * (1 - s)                 # Reset auf 0
            v_pre_hist.append(v_pre)
            v_hist.append(v)
            s_hist.append(s)

        return torch.stack(v_pre_hist, -1), torch.stack(v_hist, -1), torch.stack(s_hist, -1)

class SNNEventModel(nn.Module):
    def __init__(self, n_filters=8, kernel_size=8, tau=5.0, threshold=1.0):
        super().__init__()
        self.bank = CausalConvBank(n_filters, kernel_size)
        self.lif = LIFLayer(tau=tau, threshold=threshold)
        self.register_buffer("input_bias", torch.tensor(0.0, dtype=torch.float32))  # PI regelt hierauf

    def forward(self, x):                        # x: [B,1,L]
        I = self.bank(x) + self.input_bias
        v_pre, v, s = self.lif(I)
        logits = (v_pre - self.lif.threshold.to(v_pre.dtype)).amax(dim=1)  # [B,L]
        return {"I": I, "v_pre": v_pre, "v": v, "s": s, "logits": logits}

# =========================
# PI calibration
# =========================
@torch.no_grad()
def precalibrate_pi(model, loader, device="cpu",
                    rounds=40, max_batches=12,
                    kp=4.0, ki=0.4,
                    bias_clip=8.0, i_clip=1.0,
                    verbose=False):
    model.eval()
    integ, prev_err = 0.0, None
    tgt_rate = spk_rate = 0.0

    for r in range(rounds):
        tgt_sum = spk_sum = count = 0.0

        for b, (xb, yb) in enumerate(loader):
            if b >= max_batches:
                break
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            tgt_sum += yb.sum().item()
            spk_sum += out["s"].sum().item()     # Summe über alle LIFs
            count += yb.numel()                  # Anzahl Zeitpunkte

        tgt_rate = tgt_sum / max(count, 1.0)
        spk_rate = spk_sum / max(count, 1.0)
        err = tgt_rate - spk_rate                # spike_rate zu klein -> err > 0 -> Bias soll steigen

        # wenn Fehlerseite wechselt, altes Integral verwerfen
        if prev_err is not None and err * prev_err < 0:
            integ = 0.0

        integ_candidate = max(-i_clip, min(i_clip, integ + err))
        u = kp * err + ki * integ_candidate

        bias_old = float(model.input_bias.item())
        bias_new = bias_old + u

        # anti-windup light
        pushing_into_upper_sat = (bias_old >= bias_clip and u > 0)
        pushing_into_lower_sat = (bias_old <= -bias_clip and u < 0)
        if not (pushing_into_upper_sat or pushing_into_lower_sat):
            integ = integ_candidate

        model.input_bias.fill_(max(-bias_clip, min(bias_clip, bias_new)))
        prev_err = err

        if verbose:
            print(
                f"PI round {r:02d} | bias {float(model.input_bias.item()):.3f} | "
                f"target_rate {tgt_rate:.4f} | spike_rate {spk_rate:.4f} | err {err:.4f}"
            )

        if abs(err) < 1e-3:
            break

    return float(model.input_bias.item()), float(tgt_rate), float(spk_rate)

# =========================
# train / eval
# =========================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb)["logits"], yb).item() * xb.size(0)
        n += xb.size(0)
    return total / n

def fit(model, train_loader, test_loader, epochs=20, lr=1e-3, device="cpu", recal_every=3):
    ytr = train_loader.dataset.tensors[1]
    pos = ytr.sum().clamp_min(1.0)
    neg = ytr.numel() - pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=(neg / pos).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # Phase 1: nur PI-Vorkalibrierung
    bias, tgt_rate, spk_rate = precalibrate_pi(model, train_loader, device=device, verbose=True)
    print(f"PI pre-calibration | bias {bias:.3f} | target_rate {tgt_rate:.4f} | spike_rate {spk_rate:.4f}")

    # Phase 2: Gewichte lernen, Bias bleibt zunächst fix; gelegentlich kurz nachkalibrieren
    for ep in range(1, epochs + 1):
        if ep > 1 and recal_every and ep % recal_every == 0:
            bias, tgt_rate, spk_rate = precalibrate_pi(
                model, train_loader, device=device,
                rounds=8, max_batches=8, verbose=False
            )
        else:
            bias, tgt_rate, spk_rate = float(model.input_bias.item()), float("nan"), float("nan")

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

        te = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {ep:03d} | train {total/n:.4f} | test {te:.4f} | "
            f"bias {bias:.3f} | target_rate {tgt_rate:.4f} | spike_rate {spk_rate:.4f}"
        )

    return model

# =========================
# plot one sequence
# =========================
@torch.no_grad()
def plot_sequence(model, loader, device="cpu", idx=0):
    model.eval()
    X, Y = loader.dataset.tensors
    x = X[idx:idx+1].to(device)
    y = Y[idx].cpu()                             # [L]
    out = model(x)
    s = out["s"][0].cpu()                        # [F,L] harte Spikes

    M = torch.cat([y.unsqueeze(0), s], dim=0).numpy()
    labels = ["target"] + [f"LIF {i}" for i in range(s.size(0))]

    plt.figure(figsize=(12, 4.5))
    plt.imshow(M, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.yticks(range(len(labels)), labels)
    plt.xticks(range(M.shape[1]))
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
model = SNNEventModel(n_filters=8, kernel_size=8, tau=5.0, threshold=1.0)
model = fit(model, train_loader, test_loader, epochs=20, lr=1e-3, device=device, recal_every=3)

plot_sequence(model, test_loader, device=device, idx=0)