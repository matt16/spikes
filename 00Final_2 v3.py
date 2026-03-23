import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import math

# =========================
# data
# =========================
def make_xy(x, r, L, event_thr=-0.02):
    X = x.unfold(0, L, 1)                         # [N,2,L]
    Y = (r < event_thr).float().unfold(0, L, 1)  # [N,L]
    return X, Y

def market_data_loaders(path="market_data.xlsx", L=52, B=64, split=0.8, event_thr=-0.02):
    df = pd.read_excel(path, sheet_name="US")
    df = df.rename(columns={df.columns[0]: "Date"}).sort_values("Date")

    x = torch.tensor(df[["CF", "ED"]].iloc[1:].values, dtype=torch.float32)
    r = torch.tensor(df["_MKT"].pct_change().dropna().values, dtype=torch.float32)

    cut = int(split * len(r))
    Xtr, Ytr = make_xy(x[:cut], r[:cut], L, event_thr)
    Xte, Yte = make_xy(x[cut:], r[cut:], L, event_thr)

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
        w = w / w.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8).view(-1, 1, 1)
        return F.conv1d(F.pad(x, (self.k - 1, 0)), w)

class LIFLayer(nn.Module):
    def __init__(self, n_filters, tau=3.0, threshold=0.25, beta=15.0, dt=1.0):
        super().__init__()
        self.dt = float(dt)
        self.register_buffer("thr", torch.tensor(float(threshold), dtype=torch.float32))
        self.beta = float(beta)
        raw0 = math.log(math.expm1(float(tau)))
        self.raw_tau = nn.Parameter(torch.full((n_filters,), raw0, dtype=torch.float32))

    def tau(self):
        return F.softplus(self.raw_tau) + 1e-4

    def forward(self, I):                                # I: [B,C,L]
        B, C, L = I.shape
        v = I.new_zeros(B, C)
        z_hist, s_hist = [], []

        alpha = torch.exp(-self.dt / self.tau()).to(I.dtype).view(1, C)
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

        self.bank_a = CausalConvBank(n_filters, kernel_size)
        self.bank_b = CausalConvBank(n_filters, kernel_size)
        self.lif_a = LIFLayer(n_filters, tau=tau, threshold=threshold, beta=beta)
        self.lif_b = LIFLayer(n_filters, tau=tau, threshold=threshold, beta=beta)

        rawg0 = math.log(math.expm1(1.0))
        self.raw_gain_a = nn.Parameter(torch.tensor(rawg0, dtype=torch.float32))
        self.raw_gain_b = nn.Parameter(torch.tensor(rawg0, dtype=torch.float32))

        self.kp = float(kp)
        self.gain_min, self.gain_max = float(gain_min), float(gain_max)
        self.register_buffer("gain", torch.tensor(1.0))

    def gains(self):
        return F.softplus(self.raw_gain_a) + 1e-4, F.softplus(self.raw_gain_b) + 1e-4

    def forward(self, x):                                # x: [B,2,L]
        ga, gb = self.gains()
        Ia = self.gain * ga * self.bank_a(self.x_scale * x[:, 0:1])
        Ib = self.gain * gb * self.bank_b(self.x_scale * x[:, 1:2])

        za, sa = self.lif_a(Ia)
        zb, sb = self.lif_b(Ib)

        I = torch.cat([Ia, Ib], dim=1)
        z = torch.cat([za, zb], dim=1)
        s = torch.cat([sa, sb], dim=1)

        return {"I": I, "z": z, "s": s, "logits": z.amax(dim=1)}

    @torch.no_grad()
    def p_update_epoch(self, spike_sum, target_sum, spike_sites):
        err = target_sum / spike_sites - spike_sum / spike_sites
        self.gain.add_(self.kp * err).clamp_(self.gain_min, self.gain_max)
        return err

# =========================
# loss
# =========================
def match_1d(t, p, miss_cost=1.0, fp_cost=1.0, move_scale=2.0):
    m, n = len(t), len(p)
    D = [[j * fp_cost for j in range(n + 1)]]
    B = [[2] * (n + 1)]  # 0=match, 1=miss, 2=fp

    for i in range(1, m + 1):
        D.append([i * miss_cost] + [0.0] * n)
        B.append([1] + [0] * n)
        for j in range(1, n + 1):
            a = D[i - 1][j - 1] + abs(t[i - 1] - p[j - 1]) / move_scale
            b = D[i - 1][j] + miss_cost
            c = D[i][j - 1] + fp_cost
            D[i][j], B[i][j] = min((a, 0), (b, 1), (c, 2), key=lambda x: x[0])

    i, j = m, n
    pairs, miss, fp = [], [], []
    while i or j:
        b = B[i][j]
        if i and j and b == 0:
            pairs.append((t[i - 1], p[j - 1])); i -= 1; j -= 1
        elif i and (not j or b == 1):
            miss.append(t[i - 1]); i -= 1
        else:
            fp.append(p[j - 1]); j -= 1

    return pairs[::-1], miss[::-1], fp[::-1]

def event_loss(z, s, y, move_scale=2.0, miss_cost=1.0, fp_cost=1.0,
               lambda_birth=0.4, birth_r=1, lambda_hit=0.2, hit_r=0, lambda_dup=0.05):
    B, C, L = z.shape
    idx = torch.arange(L, device=z.device, dtype=z.dtype)
    norm = max(L - 1, 1)

    e_hard = (s.sum(dim=1) > 0).to(z.dtype)                 # [B,L] echte event-spikes
    e_soft = 1.0 - (1.0 - torch.sigmoid(z)).prod(dim=1)     # [B,L] surrogate
    e = e_hard + e_soft - e_soft.detach()                   # ST

    main = birth = hit = z.new_tensor(0.0)
    n_main = n_birth = n_hit = 0

    for b in range(B):
        t = torch.where(y[b] > 0)[0].tolist()
        p = torch.where(e_hard[b] > 0)[0].tolist()
        pairs, miss, fp = match_1d(t, p, miss_cost, fp_cost, move_scale)

        for ti, pj in pairs:
            lo, hi = max(0, pj - birth_r), min(L, pj + birth_r + 1)
            w = e_soft[b, lo:hi]
            w = w / w.sum().clamp_min(1e-6)
            main = main + (w * (idx[lo:hi] - float(ti)).abs() / norm).sum()
            n_main += 1

            lo, hi = max(0, ti - hit_r), min(L, ti + hit_r + 1)
            hit = hit - torch.log(e_soft[b, lo:hi].max().clamp_min(1e-6))
            n_hit += 1

        if miss:
            main = main + miss_cost * len(miss)
            n_main += len(miss)
        if fp:
            main = main + fp_cost * e[b, fp].sum()
            n_main += len(fp)

        for ti in miss:
            lo, hi = max(0, ti - birth_r), min(L, ti + birth_r + 1)
            birth = birth - torch.log(e_soft[b, lo:hi].max().clamp_min(1e-6))
            n_birth += 1

    u = torch.sigmoid(z).sum(dim=1)
    dup = (u - 1.0).relu().pow(2).mean()

    return (
        main / max(n_main, 1)
        + lambda_birth * birth / max(n_birth, 1)
        + lambda_hit * hit / max(n_hit, 1)
        + lambda_dup * dup
    )

# =========================
# train / eval
# =========================
@torch.no_grad()
def evaluate(model, loader, device, move_scale=2.0, miss_cost=1.0, fp_cost=1.0,
             lambda_birth=0.4, birth_r=1, lambda_hit=0.2, hit_r=0, lambda_dup=0.05):
    model.eval()
    total, n, spk, tgt = 0.0, 0, 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        e = (out["s"].sum(dim=1) > 0).float()
        total += event_loss(out["z"], out["s"], yb, move_scale, miss_cost, fp_cost,
                            lambda_birth, birth_r, lambda_hit, hit_r, lambda_dup).item() * xb.size(0)
        n += xb.size(0)
        spk += e.sum().item()
        tgt += yb.sum().item()
    return total / n, spk, tgt

def fit(model, train_loader, test_loader, epochs=20, lr=1e-3, device="cpu",
        move_scale=2.0, miss_cost=1.0, fp_cost=1.0,
        lambda_birth=0.4, birth_r=1, lambda_hit=0.2, hit_r=0, lambda_dup=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    hist = {"train": [], "test": [], "gain": [], "gain_a": [], "gain_b": []}

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        train_spk, train_tgt, train_sites = 0.0, 0.0, 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            e = (out["s"].sum(dim=1) > 0).float()
            loss = event_loss(out["z"], out["s"], yb, move_scale, miss_cost, fp_cost,
                              lambda_birth, birth_r, lambda_hit, hit_r, lambda_dup)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)
            n += xb.size(0)
            train_spk += e.sum().item()
            train_tgt += yb.sum().item()
            train_sites += e.numel()

        err = model.p_update_epoch(train_spk, train_tgt, train_sites)
        tr_loss = total / n
        te_loss, te_spk, te_tgt = evaluate(
            model, test_loader, device, move_scale, miss_cost, fp_cost,
            lambda_birth, birth_r, lambda_hit, hit_r, lambda_dup
        )

        ga, gb = model.gains()
        hist["train"].append(tr_loss)
        hist["test"].append(te_loss)
        hist["gain"].append(model.gain.item())
        hist["gain_a"].append(ga.item())
        hist["gain_b"].append(gb.item())

        print(
            f"Epoch {ep:03d} | train {tr_loss:.4f} | test {te_loss:.4f} | "
            f"gain {model.gain.item():.3f} | ga {ga.item():.3f} | gb {gb.item():.3f} | "
            f"train event-spikes {int(train_spk)} | train targets {int(train_tgt)} | "
            f"test event-spikes {int(te_spk)} | test targets {int(te_tgt)} | err {err:.5f}"
        )

    return model, hist

# =========================
# plots / diagnostics
# =========================
def plot_history(hist):
    plt.figure(figsize=(8, 4))
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
    e = (s.sum(dim=0) > 0).float()
    ga, gb = model.gains()

    labels = ["target"] + [f"CF {i}" for i in range(s.size(0)//2)] + [f"ED {i}" for i in range(s.size(0)//2)]
    M = torch.cat([y.unsqueeze(0), s], dim=0).numpy()

    plt.figure(figsize=(12, 4.5))
    plt.imshow(M, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("time step in sequence")
    plt.title(
        f"idx={idx} | gain={model.gain.item():.3f} | ga={ga.item():.3f} | gb={gb.item():.3f} | "
        f"targets={int(y.sum())} | event spikes={int(e.sum())}"
    )
    plt.tight_layout()
    plt.show()

    print(f"idx {idx:03d} | targets {int(y.sum().item())} | event spikes {int(e.sum().item())} | channel spikes {int(s.sum().item())}")
    print(f"global gain: {model.gain.item():.3f} | gain_a: {ga.item():.3f} | gain_b: {gb.item():.3f}")
    print("taus CF:", [round(t, 3) for t in model.lif_a.tau().detach().cpu().tolist()])
    print("taus ED:", [round(t, 3) for t in model.lif_b.tau().detach().cpu().tolist()])
    print("Target positions:", torch.where(y > 0)[0].tolist())
    print("Event spike positions:", torch.where(e > 0)[0].tolist())
    for i in range(s.size(0)):
        print(f"{labels[i+1]} spikes at:", torch.where(s[i] > 0)[0].tolist())

@torch.no_grad()
def sample_spike_stats(model, loader, device="cpu", n=20):
    model.eval()
    ch_counts, ev_counts, y_counts = [], [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        s = out["s"].cpu()
        e = (s.sum(dim=1) > 0).float()
        ch_counts.append(s.sum(dim=(1, 2)))
        ev_counts.append(e.sum(dim=1))
        y_counts.append(yb.sum(dim=1).cpu())

    ch_counts, ev_counts, y_counts = torch.cat(ch_counts), torch.cat(ev_counts), torch.cat(y_counts)

    print(f"Mean channel spikes/sample: {ch_counts.float().mean().item():.2f}")
    print(f"Mean event spikes/sample:   {ev_counts.float().mean().item():.2f}")
    print(f"Mean targets/sample:        {y_counts.float().mean().item():.2f}")
    print(f"Zero-event samples:         {int((ev_counts == 0).sum())}/{len(ev_counts)}")
    print(f"Zero-target samples:        {int((y_counts == 0).sum())}/{len(y_counts)}")
    print("--- first samples ---")
    for i in range(min(n, len(ev_counts))):
        print(f"idx {i:03d} | targets {int(y_counts[i].item())} | event spikes {int(ev_counts[i].item())} | channel spikes {int(ch_counts[i].item())}")
    return ch_counts, ev_counts, y_counts

def top_target_indices(loader, n=6):
    Y = loader.dataset.tensors[1]
    return torch.topk(Y.sum(dim=1), k=min(n, len(Y))).indices.tolist()

@torch.no_grad()
def plot_sequences(model, loader, device="cpu", idxs=(0, 1, 2, 3)):
    for idx in idxs:
        plot_sequence(model, loader, device=device, idx=int(idx))

# =========================
# run
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, returns = market_data_loaders(
    path="market_data.xlsx",
    L=52, B=64, split=0.8, event_thr=-0.02
)

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
    move_scale=2.0, miss_cost=1.0, fp_cost=1.0,
    lambda_birth=0.4, birth_r=1, lambda_hit=0.2, hit_r=0, lambda_dup=0.05
)

plot_history(hist)
ga, gb = model.gains()
print(f"Learned gains | global: {model.gain.item():.3f} | gain_a: {ga.item():.3f} | gain_b: {gb.item():.3f}")
print("Learned taus CF:", [round(t, 3) for t in model.lif_a.tau().detach().cpu().tolist()])
print("Learned taus ED:", [round(t, 3) for t in model.lif_b.tau().detach().cpu().tolist()])
sample_spike_stats(model, test_loader, device=device, n=12)
plot_sequences(model, test_loader, device=device, idxs=range(4))
plot_sequences(model, test_loader, device=device, idxs=top_target_indices(test_loader, n=4))