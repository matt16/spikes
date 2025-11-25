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
X = 0.5 * (np.sin(2*np.pi*10*time) + 1.0)
Y = X.copy()
#Y = (time * 0.8) + 0.05 * np.random.randn(seq_len)
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
    def __init__(self, T, K, tau=10):
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
        #psp = torch.exp(-dt / self.tau)  # (B, K, T)
        psp = torch.clamp(1.0 - dt / self.tau, min=0.0)

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

#----------------Utilities-Plotting---------
def plot_matcher_table_for_channel(model, raw_windows, lat_windows, channel_idx=0, window_idx=0):
    """
    Zeigt eine Tabelle mit:
      - Rohsignal (raw input)
      - Latencies (encodierte Spikezeiten)
      - Pro Matcher k:
          * Arrivals[k, :]
          * Delays[k, :]
          * Weights w[k, :]
          * pot[k] (nur in der Arrivals-Zeile)
    """

    # --- Rohsignal holen ---
    if isinstance(raw_windows, torch.Tensor):
        raw = raw_windows[window_idx, channel_idx, :].detach().cpu().numpy()
    else:
        raw = raw_windows[window_idx, channel_idx, :]

    # --- Latencies aus Tensor ---
    lat = lat_windows[window_idx, channel_idx, :].detach().cpu().numpy()

    # --- Matcher durchlaufen ---
    model.eval()
    with torch.no_grad():
        spike_times = lat_windows[window_idx, channel_idx, :].unsqueeze(0)  # (1, T)
        matcher = model.matchers[channel_idx]
        pot, arrivals, _ = matcher(spike_times)

        # delays & weights direkt aus dem Matcher
        delays = matcher.delays.detach().cpu().numpy()   # (K, T)
        weights = matcher.w.detach().cpu().numpy()       # (K, T)

    pot_np = pot.squeeze(0).cpu().numpy()              # (K,)
    arrivals_np = arrivals.squeeze(0).cpu().numpy()    # (K, T)

    K = pot_np.shape[0]
    T = len(lat)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    # Spaltenüberschriften: t0..t(T-1) + pot
    col_labels = [f"t{t}" for t in range(T)] + ["pot"]

    # Tabelle füllen
    table_data = []

    # Rohsignal
    raw_row = [f"{raw[t]:.3f}" for t in range(T)] + ["-"]
    table_data.append(raw_row)

    # Latencies
    lat_row = [f"{lat[t]:.2f}" for t in range(T)] + ["-"]
    table_data.append(lat_row)

    # Matcher k: arrivals, delays, weights + pot (nur bei arrivals)
    row_labels = ["raw", "latency"]

    for k in range(K):
        # Arrivals-Zeile + pot
        arr_row = [f"{arrivals_np[k, t]:.2f}" for t in range(T)] + [f"{pot_np[k]:.3f}"]
        table_data.append(arr_row)
        row_labels.append(f"arr {k}")

        # Delays-Zeile (kein pot)
        del_row = [f"{delays[k, t]:.2f}" for t in range(T)] + ["-"]
        table_data.append(del_row)
        row_labels.append(f"delay {k}")

        # Weights-Zeile (kein pot)
        w_row = [f"{weights[k, t]:.2f}" for t in range(T)] + ["-"]
        table_data.append(w_row)
        row_labels.append(f"w {k}")

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)

    ax.set_title(
        f"Kanal {channel_idx}, Fenster {window_idx}\n"
        f"Rohsignal + Latenzen + Matcher: Arrivals / Delays / Weights + Pot",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()



def plot_predicted_vs_true_latencies(model, lat_windows, channel_idx=0, window_idx=0):
    """
    Plot für ein einziges Fenster und einen Kanal:
    predicted latencies vs true latencies (T Werte).
    """

    model.eval()
    with torch.no_grad():
        preds, _ = model(lat_windows[window_idx:window_idx+1])  # shape (1, C, T)

    pred = preds[0, channel_idx].cpu().numpy()      # (T,)
    true = lat_windows[window_idx, channel_idx].cpu().numpy()  # (T,)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(true, label="true latencies", linewidth=2)
    plt.plot(pred, label="predicted latencies", linestyle="--")
    plt.title(f"Predicted vs True Latencies (Kanal {channel_idx}, Fenster {window_idx})")
    plt.xlabel("Zeitindex")
    plt.ylabel("Latency (t)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_delay_weight_history(delay_hist, weight_hist, channel_idx=0):
    """
    delay_hist: Liste von (K,T) Arrays über Epochen
    weight_hist: Liste von (K,T) Arrays über Epochen
    """
    delay_hist = np.array(delay_hist)  # shape (E, K, T)
    weight_hist = np.array(weight_hist)  # shape (E, K, T)

    E, K, T = delay_hist.shape

    import matplotlib.pyplot as plt

    # --- Delays ---
    fig, axes = plt.subplots(K, 1, figsize=(10, 2 * K), sharex=True)
    if K == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        # Für jeden t (Zeitindex) eine Kurve über Epochen
        for t in range(T):
            ax.plot(delay_hist[:, k, t], alpha=0.7)
        ax.set_title(f"Channel 0 – Matcher {k} – Delay evolution")
        ax.set_ylabel("delay")
        ax.grid(alpha=0.3)
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()

    # --- Weights ---
    fig, axes = plt.subplots(K, 1, figsize=(10, 2 * K), sharex=True)
    if K == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        for t in range(T):
            ax.plot(weight_hist[:, k, t], alpha=0.7)
        ax.set_title(f"Channel 0 – Matcher {k} – Weight evolution")
        ax.set_ylabel("weight")
        ax.grid(alpha=0.3)
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()


# -------------------- Instantiate and train --------------------
model = FullModel(channels=channels, K=K, T=T)

# --- Parameter-Gruppen: delays vs. Rest ---
delay_params = []
other_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "delays" in name:
        delay_params.append(p)
    else:
        other_params.append(p)

optimizer = optim.Adam(
    [
        {"params": other_params, "lr": 3e-3},   # normale LR
        {"params": delay_params, "lr": 1e-1},   # höhere LR nur für delays
    ]
)

mse = nn.MSELoss()

num_epochs = 500
batch_size = 128


delay_history = []   # numpy arrays shape (K, T)
weight_history = []  # numpy arrays shape (K, T)
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
        print("delay grad norm:", model.matchers[0].delays.grad)
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
    with torch.no_grad():
        d = model.matchers[0].delays.detach().cpu().numpy().copy()
        w = model.matchers[0].w.detach().cpu().numpy().copy()
        delay_history.append(d)
        weight_history.append(w)

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

plot_matcher_table_for_channel(
    model,
    windows,      # ROHINPUT
    train_lat,    # LATENZEN
    channel_idx=0,
    window_idx=5
)

plot_predicted_vs_true_latencies(model, train_lat, channel_idx=1, window_idx=5)
plot_delay_weight_history(delay_history, weight_history, channel_idx=0)




with torch.no_grad():
    preds_all, _ = model(train_lat[:6])
for i in range(6):
    print(f"Window {i}: target channel 2 pred mean {preds_all[i,2,:].mean().item():.3f} true mean {train_lat[i,2,:].mean().item():.3f}")

print("Demo complete.")