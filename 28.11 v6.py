# spiking_seq2seq_predictor_with_dendrites.py

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
Y = X.copy() + 0.005 * np.random.randn(seq_len)
#Y = (time * 0.8) + 0.05 * np.random.randn(seq_len)
shift = 3
Z = np.roll(X, shift) * 0.6 + 0.4 * (0.5*(np.tanh(3*(Y-0.5))+1.0)) + 0.03 * np.random.randn(seq_len)
series = np.stack([X, Y, Z], axis=0)  # shape (channels, seq_len)

# -------------------- Hankel windows and latency encoding --------------------
def hankel_windows_multichannel(x, T, stride=1):
    channels_, L = x.shape
    num_windows = (L - T) // stride + 1
    out = np.zeros((num_windows, channels_, T), dtype=x.dtype)
    idx = 0
    for i in range(0, L - T + 1, stride):
        out[idx] = x[:, i:i+T]
        idx += 1
    return out

windows = hankel_windows_multichannel(series, T, stride=stride)  # (num_windows, channels, T)
num_windows = windows.shape[0]

def magnitude_to_latency_batch(windows, t_min=0.0, t_max=40.0, eps=1e-8):
    nw, ch, T_ = windows.shape
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

# -------------------- Matcher: DendriticMatcherBank --------------------
class DendriticMatcherBank(nn.Module):
    def __init__(self, T, K, tau_psp=40.0, tau_mem=20.0, v_th=1.0, dt=1.0, tau_soft=1.0):
        super().__init__()
        self.T = T
        self.K = K

        # Synaptische Gewichte pro "Zeit-Synapse"
        self.w = nn.Parameter(torch.randn(K, T) * 0.2 + 0.5)

        # Delays
        self.delays = nn.Parameter(0.1 * torch.randn(K, T))

        # PSP- und Membran-Konstanten
        self.tau_psp = tau_psp
        self.tau_mem = tau_mem
        self.v_th = v_th          # aktuell nicht genutzt
        self.dt = dt

        # Temperatur für soft-argmax
        self.tau_soft = tau_soft

    def forward(self, spike_times):
        """
        spike_times: (B, T)

        Rückgabe:
            arrivals:            (B, K, T)
            psp:                 (B, K, T)
            matcher_spike_times: (B, K)
        """
        B = spike_times.size(0)
        device = spike_times.device
        T = self.T

        # 1) Arrivals
        st = spike_times.unsqueeze(1).expand(-1, self.K, -1)     # (B, K, T)
        delays = self.delays.unsqueeze(0).expand(B, -1, -1)      # (B, K, T)
        arrivals = st + delays                                   # (B, K, T)

        t_ref, _ = torch.max(arrivals, dim=2, keepdim=True)      # (B, K, 1)
        dt = torch.clamp(t_ref - arrivals, min=0.0)              # (B, K, T)

        # 2) Exponentieller PSP-Kern
        psp = torch.exp(-dt / self.tau_psp)                      # (B, K, T)

        # Synaptischer "Strom": Gewicht * PSP
        I = self.w.unsqueeze(0) * psp                            # (B, K, T)

        # 3) LIF + soft-argmax -> Spikezeit
        v = torch.zeros(B, self.K, device=device)
        v_trace = []

        for t in range(T):
            I_t = I[:, :, t]                                     # (B, K)
            dv = (-v + I_t) * (self.dt / self.tau_mem)
            v = v + dv
            v_trace.append(v.unsqueeze(-1))                      # (B, K, 1)

        v_trace = torch.cat(v_trace, dim=-1)                     # (B, K, T)

        time_axis = torch.arange(T, device=device).view(1, 1, T) * self.dt
        w_soft = torch.softmax(v_trace / self.tau_soft, dim=-1)  # (B, K, T)
        matcher_spike_times = (w_soft * time_axis).sum(dim=-1)   # (B, K)

        return arrivals, psp, matcher_spike_times


# -------------------- Vereinfachte DendriticLayer (sequentielle Verteilung) --------------------
class DendriticLayer(nn.Module):
    """
    Dendritische Layer mit einfacher sequentieller Zuordnung:

    - Wir haben n_in Eingänge (Index 0..n_in-1).
    - Wir haben n_out Ausgänge (Branches).
    - Die Eingänge werden der Reihe nach auf die Ausgänge verteilt,
      so gleichmäßig wie möglich.

      Beispiel:
        n_in = 6, n_out = 2  -> Branch 0: [0,1,2], Branch 1: [3,4,5]
        n_in = 7, n_out = 2  -> Branch 0: [0,1,2,3], Branch 1: [4,5,6]
        n_in = 3, n_out = 2  -> Branch 0: [0,1],    Branch 1: [2]

    - Intern: eine Verbindungsmaske conn_mask[o, i] ∈ {0,1}
      und eine Gewichtsmatrix W[o, i]. Nur dort, wo conn_mask = 1 ist,
      fließt Strom.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        t_max: float,
        tau_syn: float,
        tau_mem: float,
        tau_soft: float,
        dt: float = 1.0,
        mode: str = "soft_argmax",   # später evtl. "threshold"
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.t_max = t_max
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.tau_soft = tau_soft
        self.dt = dt
        self.mode = mode

        # Zeitdiskretisierung
        self.n_steps = int(round(t_max / dt))

        # --------- SEQUENTIELLE VERBINDUNGEN ----------
        # Wir teilen die n_in Indizes so gleichmäßig wie möglich auf n_out Ausgänge auf.
        #
        # base = minimale Anzahl Inputs pro Output
        # rem  = Anzahl Outputs, die EINEN zusätzlichen Input bekommen
        #
        # Beispiel n_in=7, n_out=2:
        #   base = 3, rem = 1
        #   Branch 0: 4 Inputs (0..3)
        #   Branch 1: 3 Inputs (4..6)
        conn_mask = torch.zeros(n_out, n_in, dtype=torch.float32)

        base = n_in // n_out
        rem = n_in % n_out
        cur = 0
        for o in range(n_out):
            size_o = base + (1 if o < rem else 0)
            if size_o > 0:
                conn_mask[o, cur:cur+size_o] = 1.0
                cur += size_o
            # falls size_o == 0, bleibt dieser Output rein formal „leer“

        self.register_buffer("conn_mask", conn_mask)   # (n_out, n_in)

        # Lernbare Gewichte pro mögliche Verbindung (werden mit conn_mask „ausgedünnt“)
        self.W = nn.Parameter(
            torch.ones(n_out, n_in) + 0.1 * torch.randn(n_out, n_in)
        )

    def forward(self, spike_times_in: torch.Tensor) -> torch.Tensor:
        """
        spike_times_in: (B, n_in)
        Rückgabe:
            spike_times_out: (B, n_out)
        """
        B = spike_times_in.size(0)
        device = spike_times_in.device

        # bring alles in die Form (B, n_out, n_in)
        # spike_times_expanded[b, o, i] = spike_times_in[b, i]
        spike_times_expanded = spike_times_in.unsqueeze(1).expand(B, self.n_out, self.n_in)

        v = torch.zeros(B, self.n_out, device=device)
        v_trace = []

        # conn_mask für Broadcast: (1, n_out, n_in)
        conn = self.conn_mask.view(1, self.n_out, self.n_in)

        for step in range(self.n_steps):
            t = step * self.dt

            dt_mat = t - spike_times_expanded                    # (B, n_out, n_in)
            mask_time = (dt_mat >= 0).float()

            # PSP nur dort, wo:
            #   - Spike schon passiert ist (mask_time)
            #   - eine Verbindung existiert (conn)
            psp = torch.exp(-dt_mat / self.tau_syn) * mask_time * conn

            # Gewichte in Form (1, n_out, n_in) für Broadcast
            W_eff = self.W.view(1, self.n_out, self.n_in)

            # synaptischer Input pro Branch: Summe über alle Inputs
            I = torch.sum(psp * W_eff, dim=-1)                   # (B, n_out)

            dv = (-v + I) * (self.dt / self.tau_mem)
            v = v + dv
            v_trace.append(v.unsqueeze(-1))                      # (B, n_out, 1)

        v_trace = torch.cat(v_trace, dim=-1)                     # (B, n_out, n_steps)

        if self.mode == "soft_argmax":
            time_axis = torch.arange(self.n_steps, device=device).view(1, 1, self.n_steps) * self.dt
            w = torch.softmax(v_trace / self.tau_soft, dim=-1)   # (B, n_out, n_steps)
            spike_times_out = (w * time_axis).sum(dim=-1)        # (B, n_out)
        elif self.mode == "threshold":
            raise NotImplementedError("threshold-Mode noch nicht implementiert.")
        else:
            raise ValueError(f"Unbekannter mode: {self.mode}")

        return spike_times_out


# -------------------- FullModel mit DendriticLayer & Cross-LIF --------------------
class FullModel(nn.Module):
    def __init__(self, channels, K, T, n_L2=2):
        super().__init__()
        self.channels = channels
        self.K = K
        self.T = T
        self.n_L2 = n_L2

        # Matcher-Bank pro Kanal
        self.matchers = nn.ModuleList([
            DendriticMatcherBank(T, K) for _ in range(channels)
        ])

        # Geteilte Hyperparameter für ALLE DendriticLayer
        self.dend_dt = 1.0
        self.dend_tau_syn = 5.0    # geteilt für L1, L2, Soma
        self.dend_tau_mem = 20.0   # geteilt
        self.dend_tau_soft = 1.0   # geteilt

        # Beobachtungszeiträume pro Ebene
        self.t_max_L1 = 50.0
        self.t_max_L2 = 60.0
        self.t_max_soma = 75.0

        # Dendritische Layer pro Kanal: [L1, L2, Soma] – alle mit sequentieller Zuordnung
        self.dend_layers = nn.ModuleList()
        for _ in range(channels):
            layers_for_channel = nn.ModuleList()

            # L1: K -> K/2
            layers_for_channel.append(
                DendriticLayer(
                    n_in=K,
                    n_out=K // 2,
                    t_max=self.t_max_L1,
                    tau_syn=self.dend_tau_syn,
                    tau_mem=self.dend_tau_mem,
                    tau_soft=self.dend_tau_soft,
                    dt=self.dend_dt,
                    mode="soft_argmax",
                )
            )

            # L2: (K/2) -> n_L2
            layers_for_channel.append(
                DendriticLayer(
                    n_in=K // 2,
                    n_out=n_L2,
                    t_max=self.t_max_L2,
                    tau_syn=self.dend_tau_syn,
                    tau_mem=self.dend_tau_mem,
                    tau_soft=self.dend_tau_soft,
                    dt=self.dend_dt,
                    mode="soft_argmax",
                )
            )

            # Soma: n_L2 -> 1
            layers_for_channel.append(
                DendriticLayer(
                    n_in=n_L2,
                    n_out=1,
                    t_max=self.t_max_soma,
                    tau_syn=self.dend_tau_syn,
                    tau_mem=self.dend_tau_mem,
                    tau_soft=self.dend_tau_soft,
                    dt=self.dend_dt,
                    mode="soft_argmax",   # könnte später "threshold" werden
                )
            )

            self.dend_layers.append(layers_for_channel)

        # Gates zwischen Kanälen (target x source), ohne Selbstkanten
        self.gate_logits = nn.Parameter(torch.randn(channels, channels) * 0.5)
        mask = torch.ones(channels, channels)
        for i in range(channels):
            mask[i, i] = 0.0
        self.register_buffer('gate_mask', mask)

        # Cross-Channel-LIF-Parameter (zweite Schicht)
        self.cross_dt = 1.0
        self.cross_t_max = 80.0
        self.cross_n_steps = int(round(self.cross_t_max / self.cross_dt))
        self.cross_tau_syn = 10.0
        self.cross_tau_mem = 20.0
        self.cross_threshold = 1.0  # nicht mehr für harte Schwelle genutzt
        self.cross_tau_soft = 1.0   # Temperatur für soft-argmax

    def forward(self, latencies):  # latencies: (B, channels, T)
        B = latencies.size(0)
        device = latencies.device

        # 1) Matcher + (L1, L2, Soma) pro Kanal
        soma_list = []
        for c in range(self.channels):
            # Matcher
            _, _, matcher_spikes = self.matchers[c](latencies[:, c, :])  # (B, K)

            # Dendritische Layer-Kette
            x = matcher_spikes                                # (B, K)
            x = self.dend_layers[c][0](x)                     # (B, K//2)  L1
            x = self.dend_layers[c][1](x)                     # (B, n_L2)  L2
            soma_t = self.dend_layers[c][2](x).squeeze(-1)    # (B,)       Soma
            soma_list.append(soma_t)

        soma_times = torch.stack(soma_list, dim=1)            # (B, channels)

        # 2) Gates (target x source)
        gates = torch.sigmoid(self.gate_logits) * self.gate_mask  # (C, C)

        # 3) Cross-Channel-LIF: aus Soma-Spikes der anderen Kanäle
        C = self.channels
        v = torch.zeros(B, C, device=device)
        v_trace = []

        for step in range(self.cross_n_steps):
            t = step * self.cross_dt

            dt_mat = t - soma_times         # (B, C)
            mask_psp = (dt_mat >= 0).float()
            psp = torch.exp(-dt_mat / self.cross_tau_syn) * mask_psp  # (B, C)

            # I[b, tgt] = sum_src psp[b, src] * gates[tgt, src]
            I = psp @ gates.t()                                     # (B, C)

            dv = (-v + I) * (self.cross_dt / self.cross_tau_mem)
            v = v + dv
            v_trace.append(v.unsqueeze(-1))     # (B, C, 1)

        v_trace = torch.cat(v_trace, dim=-1)    # (B, C, cross_n_steps)

        # soft-argmax über die Zeitachse
        T_cross = self.cross_n_steps
        time_axis = torch.arange(T_cross, device=device).view(1, 1, T_cross) * self.cross_dt
        w_soft = torch.softmax(v_trace / self.cross_tau_soft, dim=-1)   # (B, C, T_cross)
        recon_soma_times = (w_soft * time_axis).sum(dim=-1)             # (B, C)

        # Rückgabe:
        #  - rekonstruierte Soma-Spikezeiten
        #  - Gates
        #  - originale Soma-Spikezeiten
        return recon_soma_times, gates, soma_times


#-----------Utilities----------
def cycle_suppression_loss(model, beta_cycle=1e-3):
    """
    Penalizes symmetric connections G_ij and G_ji (2-cycles) in the gate matrix.
    """
    G = torch.sigmoid(model.gate_logits) * model.gate_mask  # (C, C)
    C = G.shape[0]

    eye = torch.eye(C, device=G.device)
    mask = 1.0 - eye  # 1 off-diagonal, 0 auf Diagonale

    sym_strength = torch.abs(G * G.T) * mask
    loss = beta_cycle * 0.5 * sym_strength.sum()
    return loss


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
        {"params": delay_params, "lr": 1},      # höhere LR nur für delays
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
        recon_soma, gates, soma_orig = model(batch_lat)

        # neue Loss: rekonstruiere Soma-Spikezeiten aus den anderen Kanälen
        loss_pred = mse(recon_soma, soma_orig)

        gate_reg = torch.mean(torch.abs(gates))
        loss_cycle = cycle_suppression_loss(model, beta_cycle=10)
        loss = loss_pred + 0.01 * gate_reg + loss_cycle

        loss.backward()
        grad = model.matchers[0].delays.grad
        if grad is not None:
            print("delay grad norm:", grad.norm().item())
        else:
            print("delay grad norm: None (kein Gradfluss bis delays)")
        optimizer.step()
        train_loss += loss.item() * batch_lat.size(0)
    train_loss /= len(train_ds)

    model.eval()
    with torch.no_grad():
        val_sum = 0.0
        for (bv,) in val_loader:
            rv, gv, sv = model(bv)
            val_sum += mse(rv, sv).item() * bv.size(0)
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
    _, final_gates, _ = model(train_lat[:32])
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
    recon_all, _, soma_all = model(train_lat[:6])
for i in range(6):
    print(f"Window {i}: channel 2 recon soma {recon_all[i,2].item():.3f} true soma {soma_all[i,2].item():.3f}")
print(X)
print("Demo complete.")
