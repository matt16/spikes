# spiking_seq2seq_predictor.py
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(precision=4, sci_mode=False)

# -------------------- Config (smaller) --------------------
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

channels = 3
T = 16
seq_len = 300   # smaller
stride = 1
K = 6   # matchers per channel (mini-neurons)
t_min, t_max = 0.0, 40.0

# -------------------- Synthetic data creation --------------------
time = np.linspace(0, 1, seq_len)
X = 0.5 * (np.sin(2*np.pi*5*time) + 1.0)
Y = (time * 0.8) + 0.05 * np.random.randn(seq_len)
shift = 3
Z = np.roll(X, shift) * 0.6 + 0.4 * (0.5*(np.tanh(3*(Y-0.5))+1.0)) + 0.03 * np.random.randn(seq_len)
series = np.stack([X, Y, Z], axis=0)  # shape (channels, seq_len)

# -------------------- Hankel windows and latency encoding --------------------
def hankel_windows_multichannel(x, T, stride=1):
    channels, L = x.shape
    num_windows = (L - T) // stride + 1
    out = np.zeros((num_windows, channels, T), dtype=x.dtype)
    idx = 0
    for i in range(0, L - T + 1, stride):
        out[idx] = x[:, i:i+T]
        idx += 1
    return out

windows = hankel_windows_multichannel(series, T, stride=stride)  # (num_windows, channels, T)
num_windows = windows.shape[0]

def magnitude_to_latency_batch(windows, t_min=0.0, t_max=40.0, eps=1e-8):
    nw, ch, T = windows.shape
    lat = np.zeros_like(windows, dtype=float)
    for c in range(ch):
        vals = windows[:, c, :].reshape(-1)
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < eps:
            vmax = vmin + eps
        norm = (windows[:, c, :] - vmin) / (vmax - vmin)
        lat[:, c, :] = t_min + (1.0 - norm) * (t_max - t_min)
    return lat

latencies = magnitude_to_latency_batch(windows, t_min=t_min, t_max=t_max)  # (num_windows, channels, T)
split = int(0.8 * num_windows)
train_lat = torch.from_numpy(latencies[:split]).float()
val_lat = torch.from_numpy(latencies[split:]).float()

# -------------------- Model components --------------------
class DendriticMatcherBank(nn.Module):
    def __init__(self, T, K, tau=3.0):
        super().__init__()
        self.T = T
        self.K = K
        self.w = nn.Parameter(torch.randn(K, T) * 0.2 + 0.5)
        init = torch.linspace(T-1, 0, T).unsqueeze(0).repeat(K,1)
        self.delays = nn.Parameter(init + 0.1 * torch.randn(K, T))
        self.tau = tau

    def forward(self, spike_times):  # spike_times: (B, T)
        B = spike_times.size(0)
        st = spike_times.unsqueeze(1).expand(-1, self.K, -1)
        delays = self.delays.unsqueeze(0).expand(B, -1, -1)
        arrivals = st + delays  # (B, K, T)
        t_ref, _ = torch.max(arrivals, dim=2, keepdim=True)  # (B, K, 1)
        dt = torch.clamp(t_ref - arrivals, min=0.0)
        psp = torch.exp(-dt / self.tau)  # (B, K, T)
        pot = torch.sum(self.w.unsqueeze(0) * psp, dim=2)  # (B, K)
        return pot, arrivals, psp

class CrossPredictor(nn.Module):
    def __init__(self, in_dim, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, T)
        )
    def forward(self, x):
        return self.net(x)

class FullModel(nn.Module):
    def __init__(self, channels, K, T):
        super().__init__()
        self.channels = channels
        self.K = K
        self.T = T
        self.matchers = nn.ModuleList([DendriticMatcherBank(T, K) for _ in range(channels)])
        self.gate_logits = nn.Parameter(torch.randn(channels, channels) * 0.5)
        mask = torch.ones(channels, channels)
        for i in range(channels):
            mask[i,i] = 0.0
        self.register_buffer('gate_mask', mask)
        self.predictors = nn.ModuleList([CrossPredictor(in_dim=K*channels, T=T) for _ in range(channels)])

    def forward(self, latencies):  # latencies: (B, channels, T)
        B = latencies.size(0)
        pots = []
        for c in range(self.channels):
            pot, _, _ = self.matchers[c](latencies[:, c, :])  # (B, K)
            pots.append(pot)
        pots_stacked = torch.stack(pots, dim=1)  # (B, channels, K)
        gates = torch.sigmoid(self.gate_logits) * self.gate_mask  # (channels, channels)
        preds = []
        for tgt in range(self.channels):
            g = gates[tgt].unsqueeze(0).unsqueeze(2)  # (1, channels, 1)
            gated = pots_stacked * g  # (B, channels, K)
            flat = gated.view(B, -1)
            pred = self.predictors[tgt](flat)  # (B, T)
            preds.append(pred)
        preds = torch.stack(preds, dim=1)  # (B, channels, T)
        return preds, gates

# -------------------- Instantiate and train --------------------
model = FullModel(channels=channels, K=K, T=T)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
mse = nn.MSELoss()
num_epochs = 40
batch_size = 128

train_ds = torch.utils.data.TensorDataset(train_lat)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_ds = torch.utils.data.TensorDataset(val_lat)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

for epoch in range(1, num_epochs+1):
    model.train()
    train_loss = 0.0
    for (batch_lat,) in train_loader:
        optimizer.zero_grad()
        preds, gates = model(batch_lat)
        loss_pred = mse(preds, batch_lat)
        gate_reg = torch.mean(torch.abs(gates))
        loss = loss_pred + 0.01 * gate_reg
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_lat.size(0)
    train_loss /= len(train_ds)
    model.eval()
    with torch.no_grad():
        val_sum = 0.0
        for (bv,) in val_loader:
            pv, gv = model(bv)
            val_sum += mse(pv, bv).item() * bv.size(0)
        val_loss = val_sum / len(val_ds)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

print("Training finished.")

with torch.no_grad():
    _, final_gates = model(train_lat[:32])
gates_np = final_gates.cpu().numpy()
print("\nLearned gates (target rows, source cols)")
print(np.round(gates_np, 3))

plt.figure(figsize=(4,4))
plt.imshow(gates_np, aspect='auto')
plt.title("Learned gates (target x source)")
plt.xlabel("source channel")
plt.ylabel("target channel")
plt.colorbar()
plt.show()

with torch.no_grad():
    preds_all, _ = model(train_lat[:6])
for i in range(6):
    print(f"Window {i}: target channel 2 pred mean {preds_all[i,2,:].mean().item():.3f} true mean {train_lat[i,2,:].mean().item():.3f}")

print("Demo complete.")