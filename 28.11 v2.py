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

# -------------------- Model components --------------------
class DendriticMatcherBank(nn.Module):
    def __init__(self, T, K, tau_psp=40.0, tau_mem=20.0, v_th=1.0, dt=1.0, tau_soft=1.0):
        super().__init__()
        self.T = T
        self.K = K

        # Synaptische Gewichte pro "Zeit-Synapse"
        self.w = nn.Parameter(torch.randn(K, T) * 0.2 + 0.5)

        # Delays wie gehabt
        self.delays = nn.Parameter(0.1 * torch.randn(K, T))

        # PSP- und Membran-Konstanten
        self.tau_psp = tau_psp    # für exp(-dt / tau_psp)
        self.tau_mem = tau_mem    # für LIF
        self.v_th = v_th          # (nicht mehr explizit genutzt, aber lass ihn da)
        self.dt = dt              # Zeitschritt (hier diskret: 1.0)

        # Temperatur für soft-argmax
        self.tau_soft = tau_soft

    def forward(self, spike_times, return_spike_times: bool = False):
        """
        spike_times: (B, T)  -- Latenzen / "Input-Spikezeiten" für dieses Fenster
        return_spike_times: wenn True, zusätzlich Matcher-Spikezeiten zurückgeben

        Rückgabe (standard):
            pot: (B, K)
            arrivals: (B, K, T)
            psp: (B, K, T)

        Rückgabe (wenn return_spike_times=True):
            pot, arrivals, psp, matcher_spike_times (B, K)
        """
        B = spike_times.size(0)
        device = spike_times.device
        T = self.T

        # -----------------------------
        # 1) Arrivals wie bisher
        # -----------------------------
        st = spike_times.unsqueeze(1).expand(-1, self.K, -1)     # (B, K, T)
        delays = self.delays.unsqueeze(0).expand(B, -1, -1)      # (B, K, T)
        arrivals = st + delays                                   # (B, K, T)

        t_ref, _ = torch.max(arrivals, dim=2, keepdim=True)      # (B, K, 1)
        dt = torch.clamp(t_ref - arrivals, min=0.0)              # (B, K, T)

        # -----------------------------
        # 2) Exponentieller PSP-Kern
        # -----------------------------
        psp = torch.exp(-dt / self.tau_psp)                      # (B, K, T)

        # Synaptischer "Strom": Gewicht * PSP
        I = self.w.unsqueeze(0) * psp                            # (B, K, T)

        # -----------------------------
        # 3) Matcher-"Potenzial" (Summe über Zeit)
        # -----------------------------
        pot = torch.sum(I, dim=2)                                # (B, K)

        # -----------------------------
        # 4) LIF + soft-argmax -> Spikezeit (optional)
        # -----------------------------
        if not return_spike_times:
            return pot, arrivals, psp

        # LIF-Integration über T-Schritte: v-Trace speichern
        v = torch.zeros(B, self.K, device=device)
        v_trace = []

        for t in range(T):
            I_t = I[:, :, t]                                     # (B, K)
            dv = (-v + I_t) * (self.dt / self.tau_mem)
            v = v + dv
            v_trace.append(v.unsqueeze(-1))                      # (B, K, 1)

        v_trace = torch.cat(v_trace, dim=-1)                     # (B, K, T)

        # soft-argmax über die Zeitachse
        time_axis = torch.arange(T, device=device).view(1, 1, T) * self.dt
        w_soft = torch.softmax(v_trace / self.tau_soft, dim=-1)  # (B, K, T)
        matcher_spike_times = (w_soft * time_axis).sum(dim=-1)   # (B, K)

        return pot, arrivals, psp, matcher_spike_times


# -------------------- Neuer 3-Level-Combiner pro Kanal --------------------
class ThreeLevelCombiner(nn.Module):
    """
    Drei-Level-Combiner für EINEN Kanal.

    Input:
        matcher_spike_times: (B, K)

    Architektur:
        L1: K -> K/2 Branches (je 2 Matcher-Spikes)
             - passive LIF (kein Threshold)
             - "Branch-Spikezeit" = soft-argmax über Membran

        L2: K/2 -> n_L2 Branches
             - jede L2-Branch liest alle L1-Branches (mit eigenen Gewichten)
             - wieder passive LIF + soft-argmax

        Soma: n_L2 -> 1 Spikezeit
             - LIF ohne harte Schwelle
             - Spikezeit = soft-argmax über v_soma(t)
    """

    def __init__(
        self,
        K: int,
        n_L2: int,
        dt: float = 1.0,
        t_max_L1: float = 50.0,
        t_max_L2: float = 60.0,
        t_max_soma: float = 75.0,
        tau_syn_L1: float = 5.0,
        tau_mem_L1: float = 20.0,
        tau_syn_L2: float = 5.0,
        tau_mem_L2: float = 20.0,
        tau_syn_soma: float = 8.0,
        tau_mem_soma: float = 20.0,
        tau_soft_L1: float = 1.0,
        tau_soft_L2: float = 1.0,
        tau_soft_soma: float = 1.0,
    ):
        super().__init__()
        assert K % 2 == 0, "K muss gerade sein, um Paare für L1 bilden zu können."
        self.K = K
        self.n_L1 = K // 2
        self.n_L2 = n_L2

        # Zeitdiskretisierung
        self.dt = dt
        self.t_max_L1 = t_max_L1
        self.t_max_L2 = t_max_L2
        self.t_max_soma = t_max_soma

        self.n_steps_L1 = int(round(t_max_L1 / dt))
        self.n_steps_L2 = int(round(t_max_L2 / dt))
        self.n_steps_soma = int(round(t_max_soma / dt))

        # Zeitkonstanten
        self.tau_syn_L1 = tau_syn_L1
        self.tau_mem_L1 = tau_mem_L1
        self.tau_syn_L2 = tau_syn_L2
        self.tau_mem_L2 = tau_mem_L2
        self.tau_syn_soma = tau_syn_soma
        self.tau_mem_soma = tau_mem_soma

        # soft-argmax Temperaturen
        self.tau_soft_L1 = tau_soft_L1
        self.tau_soft_L2 = tau_soft_L2
        self.tau_soft_soma = tau_soft_soma

        # L1-Gewichte: jede L1-Branch hat 2 Eingänge (2 Matcher)
        # Shape: (n_L1, 2)
        self.W_L1 = nn.Parameter(
            torch.ones(self.n_L1, 2) + 0.1 * torch.randn(self.n_L1, 2)
        )

        # L2-Gewichte: jede L2-Branch liest alle n_L1 L1-Branches
        # Shape: (n_L2, n_L1)
        self.W_L2 = nn.Parameter(
            torch.ones(self.n_L2, self.n_L1) + 0.1 * torch.randn(self.n_L2, self.n_L1)
        )

        # Soma-Gewichte: jede L2-Branch -> Soma
        # Shape: (n_L2,)
        self.W_soma = nn.Parameter(torch.ones(self.n_L2))

        # Matcher-Paare für L1: (0,1), (2,3), ...
        idx0 = torch.arange(0, K, 2)
        idx1 = idx0 + 1
        self.register_buffer("idx_L1_0", idx0)
        self.register_buffer("idx_L1_1", idx1)

    def _psp_exp(self, t, spike_times, tau_syn):
        dt = t - spike_times
        mask = (dt >= 0).float()
        return torch.exp(-dt / tau_syn) * mask

    def _soft_argmax_time(self, v_trace, dt, tau_soft, dim=-1):
        """
        v_trace: (..., T)
        Rückgabe: (...,)  – erwartete Zeit auf Basis softmax(v/tau_soft)
        """
        T = v_trace.size(dim)
        device = v_trace.device
        time_axis = torch.arange(T, device=device, dtype=v_trace.dtype)
        # bring time_axis in richtige Form zum Broadcasten
        while time_axis.dim() < v_trace.dim():
            time_axis = time_axis.unsqueeze(0)
        w = torch.softmax(v_trace / tau_soft, dim=dim)
        return (w * (time_axis * dt)).sum(dim=dim)

    def forward(self, matcher_spike_times):
        """
        matcher_spike_times: (B, K)

        Rückgabe:
            soma_spike_times: (B,)
            L1_spike_times:   (B, n_L1)
            L2_spike_times:   (B, n_L2)
        """
        B, K = matcher_spike_times.shape
        assert K == self.K, "K im Combiner und K im Input müssen übereinstimmen."
        device = matcher_spike_times.device

        # ================================
        # L1: K Matcher -> n_L1 Branches
        # ================================
        t0 = matcher_spike_times[:, self.idx_L1_0]  # (B, n_L1)
        t1 = matcher_spike_times[:, self.idx_L1_1]  # (B, n_L1)
        L1_inputs = torch.stack([t0, t1], dim=-1)   # (B, n_L1, 2)

        v_L1 = torch.zeros(B, self.n_L1, device=device)
        v_L1_trace = []

        for step in range(self.n_steps_L1):
            t = step * self.dt
            psp = self._psp_exp(t, L1_inputs, self.tau_syn_L1)   # (B, n_L1, 2)
            I_L1 = torch.sum(
                psp * self.W_L1.view(1, self.n_L1, 2),
                dim=-1
            )                                                   # (B, n_L1)
            dv = (-v_L1 + I_L1) * (self.dt / self.tau_mem_L1)
            v_L1 = v_L1 + dv
            v_L1_trace.append(v_L1.unsqueeze(-1))               # (B, n_L1, 1)

        v_L1_trace = torch.cat(v_L1_trace, dim=-1)              # (B, n_L1, n_steps_L1)
        L1_spike_times = self._soft_argmax_time(
            v_L1_trace, dt=self.dt, tau_soft=self.tau_soft_L1, dim=-1
        )                                                       # (B, n_L1)

        # ================================
        # L2: n_L1 Branches -> n_L2 Branches
        # ================================
        L2_inputs = L1_spike_times.unsqueeze(1).expand(B, self.n_L2, self.n_L1)

        v_L2 = torch.zeros(B, self.n_L2, device=device)
        v_L2_trace = []

        for step in range(self.n_steps_L2):
            t = step * self.dt
            psp = self._psp_exp(t, L2_inputs, self.tau_syn_L2)      # (B, n_L2, n_L1)
            I_L2 = torch.sum(
                psp * self.W_L2.view(1, self.n_L2, self.n_L1),
                dim=-1
            )                                                       # (B, n_L2)
            dv = (-v_L2 + I_L2) * (self.dt / self.tau_mem_L2)
            v_L2 = v_L2 + dv
            v_L2_trace.append(v_L2.unsqueeze(-1))                   # (B, n_L2, 1)

        v_L2_trace = torch.cat(v_L2_trace, dim=-1)                  # (B, n_L2, n_steps_L2)
        L2_spike_times = self._soft_argmax_time(
            v_L2_trace, dt=self.dt, tau_soft=self.tau_soft_L2, dim=-1
        )                                                           # (B, n_L2)

        # ================================
        # Soma: n_L2 -> 1 Spikezeit via soft-argmax
        # ================================
        v_soma = torch.zeros(B, device=device)
        v_soma_trace = []

        soma_inputs = L2_spike_times  # (B, n_L2)

        for step in range(self.n_steps_soma):
            t = step * self.dt
            dt_mat = t - soma_inputs                                 # (B, n_L2)
            mask = (dt_mat >= 0).float()
            psp = torch.exp(-dt_mat / self.tau_syn_soma) * mask      # (B, n_L2)
            I_soma = torch.sum(
                psp * self.W_soma.view(1, self.n_L2),
                dim=-1
            )                                                       # (B,)
            dv = (-v_soma + I_soma) * (self.dt / self.tau_mem_soma)
            v_soma = v_soma + dv
            v_soma_trace.append(v_soma.unsqueeze(-1))               # (B, 1)

        v_soma_trace = torch.cat(v_soma_trace, dim=-1)              # (B, n_steps_soma)
        soma_spike_times = self._soft_argmax_time(
            v_soma_trace, dt=self.dt, tau_soft=self.tau_soft_soma, dim=-1
        )                                                           # (B,)

        return soma_spike_times, L1_spike_times, L2_spike_times


# -------------------- FullModel mit neuem Combiner & Cross-LIF --------------------
class FullModel(nn.Module):
    def __init__(self, channels, K, T, n_L2=2):
        super().__init__()
        self.channels = channels
        self.K = K
        self.T = T

        # Matcher-Bank pro Kanal
        self.matchers = nn.ModuleList([
            DendriticMatcherBank(T, K) for _ in range(channels)
        ])

        # Neuer dendritischer 3-Level-Combiner pro Kanal
        self.combiners = nn.ModuleList([
            ThreeLevelCombiner(K=K, n_L2=n_L2) for _ in range(channels)
        ])

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

        # 1) Matcher + Combiner pro Kanal: Soma-Spikezeit lokal
        soma_list = []
        for c in range(self.channels):
            pot, _, _, matcher_spikes = self.matchers[c](
                latencies[:, c, :],
                return_spike_times=True
            )  # matcher_spikes: (B, K)

            soma_t, _, _ = self.combiners[c](matcher_spikes)  # (B,)
            soma_list.append(soma_t)

        soma_times = torch.stack(soma_list, dim=1)  # (B, channels)

        # 2) Gates (target x source)
        gates = torch.sigmoid(self.gate_logits) * self.gate_mask  # (C, C)

        # 3) Cross-Channel-LIF: aus Soma-Spikes der anderen Kanäle
        #    die Soma-Spikezeit des Zielkanals rekonstruieren (soft-argmax)
        C = self.channels
        v = torch.zeros(B, C, device=device)
        v_trace = []

        for step in range(self.cross_n_steps):
            t = step * self.cross_dt

            dt_mat = t - soma_times         # (B, C)
            mask_psp = (dt_mat >= 0).float()
            psp = torch.exp(-dt_mat / self.cross_tau_syn) * mask_psp  # (B, C)

            # I[b, tgt] = sum_src psp[b, src] * gates[tgt, src]
            I = psp @ gates.t()   # (B, C)

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
        #  - rekonstruierte Soma-Spikezeiten aus Cross-Channel-LIF
        #  - Gates
        #  - originale Soma-Spikezeiten (für Loss / Debug)
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


#----------------Utilities-Plotting---------
def plot_arrivals_for_window(model, lat_windows, channel_idx=0, window_idx=5):
    """
    Für ein Fenster:
      - arrivals pro Matcher
      - spike_times (Input)
      - psp pro Matcher (eigene y-Achse, da viel kleiner)
    """
    model.eval()
    with torch.no_grad():
        spike_times = lat_windows[window_idx, channel_idx, :].unsqueeze(0)  # (1, T)
        matcher = model.matchers[channel_idx]
        pot, arrivals, psp = matcher(spike_times)  # arrivals: (1, K, T), psp: (1, K, T)

    arr = arrivals.squeeze(0).cpu().numpy()   # (K, T)
    psp_np = psp.squeeze(0).cpu().numpy()     # (K, T)
    spike_np = spike_times.squeeze(0).cpu().numpy()  # (T,)

    K_, T_ = arr.shape
    t_idx = np.arange(T_)

    fig, axes = plt.subplots(K_, 1, figsize=(8, 3 * K_), sharex=True)
    if K_ == 1:
        axes = [axes]

    for k in range(K_):
        ax = axes[k]

        arrivals_k = arr[k, :]
        psp_k = psp_np[k, :]

        # --- LEFT AXIS: arrivals + spike_times ---
        ax.scatter(t_idx, arrivals_k, s=30, label="arrival(t)")
        ax.plot(t_idx, arrivals_k, alpha=0.6)

        # spike_times
        ax.plot(t_idx, spike_np, linestyle="--", linewidth=1.5,
                label="spike_times")

        # mean arrival
        mean_val = arrivals_k.mean()
        ax.axhline(mean_val, linestyle=":", linewidth=2,
                   label=f"mean={mean_val:.2f}")

        ax.set_ylabel("time")
        ax.set_title(f"Channel {channel_idx}, Window {window_idx} – Matcher {k}")
        ax.grid(alpha=0.3)

        # --- RIGHT AXIS: PSP values ---
        ax2 = ax.twinx()
        ax2.plot(t_idx, psp_k, color="tab:red", linewidth=1.5,
                 label="psp(t)")
        ax2.set_ylabel("psp", color="tab:red")
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(0, max(psp_k)*1.2 + 1e-3)  # etwas Luft nach oben

        # Combined Legend (left + right sides)
        lines_left, labels_left = ax.get_legend_handles_labels()
        lines_right, labels_right = ax2.get_legend_handles_labels()
        ax.legend(lines_left + lines_right, labels_left + labels_right,
                  loc="upper right")

    axes[-1].set_xlabel("time index")
    plt.tight_layout()
    plt.show()


def plot_matcher_table_for_channel(model, raw_windows, lat_windows, channel_idx=0, window_idx=0):
    """
    Zeigt eine Tabelle mit:
      - Rohsignal (raw input)
      - Latencies (encodierte Spikezeiten)
      - Pro Matcher k:
          * Arrivals[k, :]
          * Delays[k, :]
          * Weights w[k, :]
          * pot[k] (nur bei arrivals)
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

    K_ = pot_np.shape[0]
    T_ = len(lat)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    # Spaltenüberschriften: t0..t(T-1) + pot
    col_labels = [f"t{t}" for t in range(T_)] + ["pot"]

    # Tabelle füllen
    table_data = []

    # Rohsignal
    raw_row = [f"{raw[t]:.3f}" for t in range(T_)] + ["-"]
    table_data.append(raw_row)

    # Latencies
    lat_row = [f"{lat[t]:.2f}" for t in range(T_)] + ["-"]
    table_data.append(lat_row)

    # Matcher k: arrivals, delays, weights + pot (nur bei arrivals)
    row_labels = ["raw", "latency"]

    for k in range(K_):
        # Arrivals-Zeile + pot
        arr_row = [f"{arrivals_np[k, t]:.2f}" for t in range(T_)] + [f"{pot_np[k]:.3f}"]
        table_data.append(arr_row)
        row_labels.append(f"arr {k}")

        # Delays-Zeile (kein pot)
        del_row = [f"{delays[k, t]:.2f}" for t in range(T_)] + ["-"]
        table_data.append(del_row)
        row_labels.append(f"delay {k}")

        # Weights-Zeile (kein pot)
        w_row = [f"{weights[k, t]:.2f}" for t in range(T_)] + ["-"]
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
    Angepasst: wir plotten jetzt "true vs reconstructed" Soma-Spikezeit
    für EIN Fenster und EINEN Kanal.
    """
    model.eval()
    with torch.no_grad():
        recon_soma, _, soma_orig = model(lat_windows[window_idx:window_idx+1])  # (1, C), (1, C)

    recon_val = recon_soma[0, channel_idx].item()
    true_val = soma_orig[0, channel_idx].item()

    plt.figure(figsize=(6,4))
    plt.bar(["true_soma_t", "recon_soma_t"], [true_val, recon_val])
    plt.title(f"Soma Spike Time: true vs recon (Channel {channel_idx}, Window {window_idx})")
    plt.ylabel("Spike time")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_delay_weight_history(delay_hist, weight_hist, channel_idx=0):
    """
    delay_hist: Liste von (K,T) Arrays über Epochen
    weight_hist: Liste von (K,T) Arrays über Epochen
    """
    delay_hist = np.array(delay_hist)  # shape (E, K, T)
    weight_hist = np.array(weight_hist)  # shape (E, K, T)

    E, K_, T_ = delay_hist.shape

    # --- Delays ---
    fig, axes = plt.subplots(K_, 1, figsize=(10, 2 * K_), sharex=True)
    if K_ == 1:
        axes = [axes]
    for k in range(K_):
        ax = axes[k]
        for t in range(T_):
            ax.plot(delay_hist[:, k, t], alpha=0.7)
        ax.set_title(f"Channel {channel_idx} – Matcher {k} – Delay evolution")
        ax.set_ylabel("delay")
        ax.grid(alpha=0.3)
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()

    # --- Weights ---
    fig, axes = plt.subplots(K_, 1, figsize=(10, 2 * K_), sharex=True)
    if K_ == 1:
        axes = [axes]
    for k in range(K_):
        ax = axes[k]
        for t in range(T_):
            ax.plot(weight_hist[:, k, t], alpha=0.7)
        ax.set_title(f"Channel {channel_idx} – Matcher {k} – Weight evolution")
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

plot_matcher_table_for_channel(
    model,
    windows,      # ROHINPUT
    train_lat,    # LATENZEN
    channel_idx=0,
    window_idx=5
)

plot_predicted_vs_true_latencies(model, train_lat, channel_idx=1, window_idx=5)
plot_delay_weight_history(delay_history, weight_history, channel_idx=0)
plot_arrivals_for_window(model, train_lat, channel_idx=0, window_idx=5)

with torch.no_grad():
    recon_all, _, soma_all = model(train_lat[:6])
for i in range(6):
    print(f"Window {i}: channel 2 recon soma {recon_all[i,2].item():.3f} true soma {soma_all[i,2].item():.3f}")
print(X)
print("Demo complete.")
