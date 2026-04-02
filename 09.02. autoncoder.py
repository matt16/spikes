# ============================================================
# CNN-AE -> Delta-Coding -> LIF Matched-Filter Bank (Demo)
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# Repro / Device
# -----------------------------
np.random.seed(7)
torch.manual_seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1) Synthetic motifs + series
# -----------------------------
def motif_library(L=48):
    t = np.linspace(0, 1, L)
    # Motif 0: ramp (trend burst)
    m0 = (t - t.mean())
    # Motif 1: oscillation burst
    m1 = np.sin(2*np.pi*5*t) * np.hanning(L)
    # Motif 2: high-freq volatility burst
    m2 = (np.sin(2*np.pi*14*t) + 0.5*np.sin(2*np.pi*21*t)) * np.hanning(L)
    # Motif 3: jump + decay
    m3 = np.zeros(L)
    m3[0] = 2.5
    m3 += 0.6*np.exp(-8*t)
    m3 -= m3.mean()
    M = np.stack([m0, m1, m2, m3], axis=0)
    M = M / (np.std(M, axis=1, keepdims=True) + 1e-8)
    return M.astype(np.float32)

MOTIFS = motif_library(L=48)
K = MOTIFS.shape[0]
MOTIF_LEN = MOTIFS.shape[1]

def generate_series(T=2048, p_insert=0.02, noise_std=0.45):
    # baseline AR(1) noise-ish
    x = np.zeros(T, dtype=np.float32)
    eps = (np.random.randn(T) * noise_std).astype(np.float32)
    for t in range(1, T):
        x[t] = 0.88 * x[t-1] + eps[t]

    # labels: 1 in a small window around "onset detection time"
    labels = np.zeros((K, T), dtype=np.float32)

    occupied = np.zeros(T, dtype=bool)
    for t0 in range(0, T - MOTIF_LEN):
        if np.random.rand() < p_insert and not occupied[t0:t0+MOTIF_LEN].any():
            k = np.random.randint(0, K)
            amp = np.random.uniform(0.8, 1.6)
            x[t0:t0+MOTIF_LEN] += amp * MOTIFS[k]
            occupied[t0:t0+MOTIF_LEN] = True

            # mark "detection target" near early part of motif
            center = t0 + MOTIF_LEN // 4
            labels[k, max(0, center-2):min(T, center+3)] = 1.0

    # normalize
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x, labels

class WindowDataset(Dataset):
    def __init__(self, n_series=200, T=2048, win=256, step=64):
        self.X = []
        self.Y = []
        for _ in range(n_series):
            x, y = generate_series(T=T)
            for s in range(0, T-win+1, step):
                self.X.append(x[s:s+win][None, :])   # [1, win]
                self.Y.append(y[:, s:s+win])         # [K, win]
        self.X = np.stack(self.X).astype(np.float32)
        self.Y = np.stack(self.Y).astype(np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

# -----------------------------
# 2) CNN Autoencoder (TCN-ish)
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.c1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.c2 = nn.Conv1d(32, 64, kernel_size=7, padding=6, dilation=2)
        self.c3 = nn.Conv1d(64, 64, kernel_size=7, padding=12, dilation=4)
        self.toz = nn.Conv1d(64, z_dim, kernel_size=1)

    def forward(self, x):
        x = F.gelu(self.c1(x))
        x = F.gelu(self.c2(x))
        x = F.gelu(self.c3(x))
        return self.toz(x)  # [B, z_dim, T]

class Decoder(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.c1 = nn.Conv1d(z_dim, 64, kernel_size=1)
        self.c2 = nn.Conv1d(64, 64, kernel_size=7, padding=12, dilation=4)
        self.c3 = nn.Conv1d(64, 32, kernel_size=7, padding=6, dilation=2)
        self.out = nn.Conv1d(32, 1, kernel_size=7, padding=3)

    def forward(self, z):
        x = F.gelu(self.c1(z))
        x = F.gelu(self.c2(x))
        x = F.gelu(self.c3(x))
        return self.out(x)

class AutoEncoder(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat, z

def ae_loss(x, xhat, z, lam_sparse=1e-3, lam_slow=2e-3):
    # L1 rec (robuster als MSE), plus sparsity & slow features
    rec = (x - xhat).abs().mean()
    sparse = z.abs().mean()
    dz = (z[:, :, 1:] - z[:, :, :-1]).abs().mean()
    return rec + lam_sparse*sparse + lam_slow*dz, {"rec": rec.item(), "sparse": sparse.item(), "slow": dz.item()}

# -----------------------------
# 3) Delta-coding (Option B)
# -----------------------------
def delta_code(z, thr=0.35):
    """
    z: [B, C, T]
    returns spikes_in: [B, 2*C, T]  (ON + OFF channels)
    """
    dz = torch.zeros_like(z)
    dz[:, :, 1:] = z[:, :, 1:] - z[:, :, :-1]

    on  = (dz >  thr).float()
    off = (dz < -thr).float()
    return torch.cat([on, off], dim=1), dz

# -----------------------------
# 4) LIF Matched Filter Bank
#    - Each neuron aims to spike for one motif k
# -----------------------------
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # hard threshold in forward
        out = (x > 0).float()
        ctx.save_for_backward(x)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # surrogate gradient (fast sigmoid derivative)
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + torch.abs(x))**2
        return grad_output * sg

spike_fn = SurrogateSpike.apply

class LIFMatchedBank(nn.Module):
    """
    Input: spike-like channels [B, Din, T]
    Output: spikes [B, K, T], membrane V [B, K, T]
    """
    def __init__(self, Din, K, tau_ms=None):
        super().__init__()
        self.Din = Din
        self.K = K
        # "Matched filter" weights: each neuron has its own weight vector on Din
        self.W = nn.Parameter(0.1 * torch.randn(K, Din))
        self.bias = nn.Parameter(torch.zeros(K))
        # thresholds can be learned too
        self.vth = nn.Parameter(torch.ones(K) * 1.0)

        # different membrane time constants => different temporal integration
        if tau_ms is None:
            tau_ms = torch.linspace(8.0, 25.0, K)  # ms-ish scale (relative)
        self.register_buffer("tau", tau_ms)

    def forward(self, x_in):
        # x_in: [B, Din, T]
        B, Din, T = x_in.shape
        assert Din == self.Din

        V = torch.zeros(B, self.K, device=x_in.device)
        V_hist = []
        S_hist = []

        # discrete-time LIF
        # V[t] = alpha * V[t-1] + I[t] - S[t-1]*vth (reset-by-subtraction)
        # alpha = exp(-1/tau)
        alpha = torch.exp(-1.0 / self.tau).unsqueeze(0)  # [1, K]

        for t in range(T):
            xt = x_in[:, :, t]  # [B, Din]
            # current: linear "matched filter" on input channels
            I = F.linear(xt, self.W, self.bias)  # [B, K]

            V = alpha * V + I
            S = spike_fn(V - self.vth)   # spike if above threshold
            V = V - S * self.vth         # reset by subtraction

            V_hist.append(V.unsqueeze(-1))
            S_hist.append(S.unsqueeze(-1))

        V_hist = torch.cat(V_hist, dim=-1)
        S_hist = torch.cat(S_hist, dim=-1)
        return S_hist, V_hist

# -----------------------------
# 5) Training loops
# -----------------------------
def train_ae(ae, loader, epochs=10, lr=2e-3):
    ae = ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for ep in range(epochs):
        ae.train()
        stats = {"loss":0, "rec":0, "sparse":0, "slow":0, "n":0}
        for xb, _ in loader:
            xb = xb.to(device)
            xhat, z = ae(xb)
            loss, parts = ae_loss(xb, xhat, z)
            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = xb.size(0)
            stats["loss"] += loss.item()*bs
            stats["rec"]  += parts["rec"]*bs
            stats["sparse"] += parts["sparse"]*bs
            stats["slow"] += parts["slow"]*bs
            stats["n"] += bs

        for k in ["loss","rec","sparse","slow"]:
            stats[k] /= stats["n"]
        print(f"AE ep {ep+1:02d}: loss {stats['loss']:.4f} | rec {stats['rec']:.4f} | sparse {stats['sparse']:.4f} | slow {stats['slow']:.4f}")
    return ae

def train_lif(bank, ae, loader, epochs=8, lr=2e-3, delta_thr=0.35):
    # freeze AE encoder
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    bank = bank.to(device)
    opt = torch.optim.Adam(bank.parameters(), lr=lr)

    bce = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        bank.train()
        tot=0.0; n=0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)  # [B, K, T]

            with torch.no_grad():
                _, z = ae(xb)
                xin, _ = delta_code(z, thr=delta_thr)  # [B, 2*z_dim, T]

            spikes, V = bank(xin)  # spikes [B,K,T], V [B,K,T]

            # We want each neuron k to spike at motif times.
            # Use membrane potential V as "logit" signal (continuous), supervise against yb.
            loss = bce(V, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item()*xb.size(0)
            n += xb.size(0)

        print(f"LIF ep {ep+1:02d}: BCE {tot/n:.4f}")

    return bank

# -----------------------------
# 6) Run the demo
# -----------------------------
if __name__ == "__main__":
    # Data
    train_ds = WindowDataset(n_series=120, T=2048, win=256, step=64)
    val_ds   = WindowDataset(n_series=20,  T=2048, win=256, step=64)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    # AE
    z_dim = 16
    ae = AutoEncoder(z_dim=z_dim)
    ae = train_ae(ae, train_loader, epochs=10, lr=2e-3)

    # LIF bank (Din = 2*z_dim due to ON/OFF channels)
    bank = LIFMatchedBank(Din=2*z_dim, K=K)
    bank = train_lif(bank, ae, train_loader, epochs=8, lr=2e-3, delta_thr=0.35)

    # -----------------------------
    # 7) Interpretation Plots on ONE fresh long series
    # -----------------------------
    T = 2048
    x, y = generate_series(T=T, p_insert=0.02)
    x_t = torch.from_numpy(x[None, None, :]).to(device)  # [1,1,T]
    y_t = torch.from_numpy(y[None, :, :]).to(device)     # [1,K,T]

    ae.eval(); bank.eval()
    with torch.no_grad():
        xhat, z = ae(x_t)
        xin, dz = delta_code(z, thr=0.35)
        spikes, V = bank(xin)

    xhat = xhat.squeeze().cpu().numpy()
    z_np = z.squeeze().cpu().numpy()         # [z_dim, T]
    dz_np = dz.squeeze().cpu().numpy()       # [z_dim, T]
    xin_np = xin.squeeze().cpu().numpy()     # [2*z_dim, T]
    spikes_np = spikes.squeeze().cpu().numpy()  # [K, T]
    V_np = V.squeeze().cpu().numpy()           # [K, T]
    y_np = y

    # Plot 1: raw vs recon + motif onsets
    plt.figure()
    plt.title("Synthetic series: raw x(t) and AE reconstruction x_hat(t)")
    plt.plot(x, linewidth=1)
    plt.plot(xhat, linewidth=1)
    for k in range(K):
        idx = np.where(y_np[k] > 0.5)[0]
        if len(idx) > 0:
            plt.scatter(idx, np.full_like(idx, np.max(x)*0.9), s=10)
    plt.xlabel("t")
    plt.ylabel("value")
    plt.tight_layout()

    # Plot 2: a few latent channels z(t)
    plt.figure()
    plt.title("Latent representation z(t) (first 6 channels)")
    for i in range(min(6, z_dim)):
        plt.plot(z_np[i], linewidth=1)
    plt.xlabel("t")
    plt.ylabel("z")
    plt.tight_layout()

    # Plot 3: delta z(t) (first 6 channels)
    plt.figure()
    plt.title("Delta latent Δz(t) (first 6 channels)")
    for i in range(min(6, z_dim)):
        plt.plot(dz_np[i], linewidth=1)
    plt.xlabel("t")
    plt.ylabel("Δz")
    plt.tight_layout()

    # Plot 4: delta-coded ON/OFF events (sum over channels as quick view)
    plt.figure()
    plt.title("Delta-coded events: ON/OFF activity (summed over channels)")
    on_sum  = xin_np[:z_dim].sum(axis=0)
    off_sum = xin_np[z_dim:].sum(axis=0)
    plt.plot(on_sum, linewidth=1)
    plt.plot(off_sum, linewidth=1)
    plt.xlabel("t")
    plt.ylabel("event count")
    plt.tight_layout()

    # Plot 5: LIF membrane potentials V_k(t) for each motif neuron
    plt.figure()
    plt.title("LIF matched filter bank: membrane potentials V_k(t)")
    for k in range(K):
        plt.plot(V_np[k], linewidth=1)
    # mark true targets per motif
    for k in range(K):
        idx = np.where(y_np[k] > 0.5)[0]
        if len(idx) > 0:
            plt.scatter(idx, np.full_like(idx, np.max(V_np[k])*0.8), s=10)
    plt.xlabel("t")
    plt.ylabel("V")
    plt.tight_layout()

    # Plot 6: LIF spikes per neuron (binary)
    plt.figure()
    plt.title("LIF spikes S_k(t) (each neuron aims to detect one motif)")
    # offset each spike train for readability
    for k in range(K):
        plt.plot(spikes_np[k] + 1.5*k, linewidth=1)
    # mark true onsets per motif as dots at the same offsets
    for k in range(K):
        idx = np.where(y_np[k] > 0.5)[0]
        if len(idx) > 0:
            plt.scatter(idx, np.full_like(idx, 1.5*k), s=10)
    plt.xlabel("t")
    plt.ylabel("spikes (offset per neuron)")
    plt.tight_layout()

    # Plot 7: learned "matched filter" weights (per neuron over input channels)
    plt.figure()
    plt.title("Matched filters: weight vectors W_k over delta-coded channels")
    W = bank.W.detach().cpu().numpy()  # [K, 2*z_dim]
    for k in range(K):
        plt.plot(W[k], linewidth=1)
    plt.xlabel("channel (ON then OFF)")
    plt.ylabel("weight")
    plt.tight_layout()

    plt.show()
