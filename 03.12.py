import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
#  Sliding window (Hankel). Each row is a pattern candidate.
# -------------------------------------------------------------
def hankel_windows(x, W):
    T = x.shape[0]
    return torch.stack([x[i-W+1 : i+1] for i in range(W-1, T)], dim=0)


# -------------------------------------------------------------
#  Simple dendrite layer: nonlinear subunit
# -------------------------------------------------------------
class DendriteLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Du hast 0.2 gut gefunden -> übernehmen wir
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.2)

    def forward(self, x):
        # Local dendritic nonlinearity
        return F.relu(x @ self.W.t())


# -------------------------------------------------------------
#  Soma ODE: membrane solves dV/dt = (-V + I) / tau
#  Returns latency = time to reach threshold or None.
# -------------------------------------------------------------
class SomaODE(nn.Module):
    def __init__(self, dt=1e-2, tau=0.05):
        super().__init__()
        self.dt = dt
        self.tau = tau
        self.V_rest = 0.0
        self.V_th   = 1.0

    def forward(self, I):
        V = self.V_rest
        t = 0.0
        # Pattern-recognition time unfolds here (processing time)
        for _ in range(200000):
            dV = (-V + I) / self.tau
            V += self.dt * dV
            t += self.dt
            if V >= self.V_th:
                return t  # latency = pattern-recognition time
        return None


# -------------------------------------------------------------
#  Full deep-dendritic encoder
# -------------------------------------------------------------
class DeepDendriticEncoder(nn.Module):
    def __init__(self, W, K):
        super().__init__()
        self.W = W

        # Three dendritic layers → “deep dendrite”
        self.d1 = DendriteLayer(W,    K)
        self.d2 = DendriteLayer(K,    K//2)
        self.d3 = DendriteLayer(K//2, K//4)

        # Soma integrates strongest dendritic current
        self.soma = SomaODE()

    def forward(self, x):
        # 1) Generate all sliding windows
        X = hankel_windows(x, self.W)      # (N, W)

        # 2) Deep dendritic computation
        h1 = self.d1(X)                    # (N, K)
        h2 = self.d2(h1)                   # (N, K/2)
        h3 = self.d3(h2)                   # (N, K/4)

        # 3) Winner dendrite = the one giving max current
        I, winner = h3.max(dim=1)          # (N,)
        # leichter Gain, damit Soma eher mal spikt
        I = I * 2.0

        # 4) Compute ODE-based latency for each window
        latencies = [self.soma(i.item()) for i in I]

        # 5) Convert recognition time into actual time:
        #    absolute_spike_time = window_start + latency
        absolute_times = [
            t_in + (lat if lat is not None else float('inf'))
            for t_in, lat in enumerate(latencies)
        ]

        # 6) Which input pattern (window index) won?
        #    (the window with the earliest absolute spike)
        best_window = torch.argmin(torch.tensor(absolute_times))

        return {
            "pattern_index"  : int(best_window),        # which input window matched
            "winner_dendrite": winner[best_window].item(),
            "latency"        : latencies[best_window],  # recognition delay (oder None)
            "abs_spike_time" : absolute_times[best_window]
        }


# -------------------------------------------------------------
#  Analyse-Funktion: gibt interne Größen zurück
# -------------------------------------------------------------
def analyze_encoder(model, x):
    """
    Macht das gleiche wie forward, gibt aber X, h1, h2, h3, I, winner,
    latencies und absolute_times zurück, damit wir plotten können.
    """
    X = hankel_windows(x, model.W)      # (N, W)

    h1 = model.d1(X)
    h2 = model.d2(h1)
    h3 = model.d3(h2)

    I, winner = h3.max(dim=1)
    I_gain = I * 2.0  # selbe Verstärkung wie im forward

    latencies = [model.soma(i.item()) for i in I_gain]
    absolute_times = [
        t_in + (lat if lat is not None else float('inf'))
        for t_in, lat in enumerate(latencies)
    ]

    return {
        "X": X,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "I": I_gain,
        "winner": winner,
        "latencies": latencies,
        "absolute_times": absolute_times,
    }


# -------------------------------------------------------------
#  Plot-Funktionen
# -------------------------------------------------------------
def plot_signal_with_best_window(x_np, pattern_index, W):
    T = len(x_np)
    t = np.arange(T)

    plt.figure()
    plt.plot(t, x_np, label="Input signal")
    start = pattern_index
    end = pattern_index + W
    plt.axvspan(start, end, alpha=0.3, label="Best window")
    plt.xlabel("time index")
    plt.ylabel("x")
    plt.legend()
    plt.title("Signal with best-matching window")
    plt.tight_layout()
    plt.show()


def plot_I_and_latencies(I_tensor, latencies):
    I_np = I_tensor.detach().cpu().numpy()
    lat_np = np.array([l if l is not None else np.nan for l in latencies])
    idx = np.arange(len(I_np))

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.stem(idx, I_np, use_line_collection=True)
    plt.ylabel("I (winner current)")
    plt.title("Winner dendritic current per window")

    plt.subplot(2, 1, 2)
    plt.stem(idx, lat_np, use_line_collection=True)
    plt.xlabel("window index")
    plt.ylabel("latency")
    plt.title("Latency per window (NaN = no spike)")

    plt.tight_layout()
    plt.show()


def soma_trace(model, I, T_steps=200, dt=None):
    if dt is None:
        dt = model.soma.dt
    V = model.soma.V_rest
    Vs = []
    ts = []
    t = 0.0
    for _ in range(T_steps):
        dV = (-V + I) / model.soma.tau
        V += dt * dV
        t += dt
        Vs.append(V)
        ts.append(t)
    return np.array(ts), np.array(Vs)


def plot_soma_traces_for_windows(model, I_tensor, window_indices):
    plt.figure()
    for idx in window_indices:
        I_val = I_tensor[idx].item()
        ts, Vs = soma_trace(model, I_val)
        plt.plot(ts, Vs, label=f"window {idx}, I={I_val:.2f}")
    plt.axhline(model.soma.V_th, linestyle="--", label="threshold")
    plt.xlabel("soma time")
    plt.ylabel("V")
    plt.legend()
    plt.title("Soma membrane traces for selected windows")
    plt.tight_layout()
    plt.show()


def plot_dendritic_activations_for_window(diag, window_idx):
    h1 = diag["h1"][window_idx].detach().cpu().numpy()
    h2 = diag["h2"][window_idx].detach().cpu().numpy()
    h3 = diag["h3"][window_idx].detach().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(np.arange(len(h1)), h1)
    plt.title(f"h1 (window {window_idx})")

    plt.subplot(1, 3, 2)
    plt.bar(np.arange(len(h2)), h2)
    plt.title("h2")

    plt.subplot(1, 3, 3)
    plt.bar(np.arange(len(h3)), h3)
    plt.title("h3")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
#  MAIN – Dummy Input + Plots
# -------------------------------------------------------------
if __name__ == "__main__":

    # 1) Dummy Input erzeugen
    T = 80
    time = np.linspace(0, 1, T)

    # Einfaches Testsignal:
    x_np = 0.5 * (np.sin(2*np.pi*4*time) + 1.0)
    x_np[30:33] += 0.8     # künstliches lokales Pattern

    x = torch.tensor(x_np, dtype=torch.float32)

    print("Input shape:", x.shape)
    print("Example input slice:", x[:10])

    # 2) Modell initialisieren
    W = 16   # Fenstergröße
    K = 32   # dendritische Units im 1. Layer

    model = DeepDendriticEncoder(W=W, K=K)

    # 3) Forward ausführen
    result = model(x)

    print("\n===== Deep Dendritic Encoder Output =====")
    print("Best-matching window index :", result["pattern_index"])
    print("Winner dendrite (index)     :", result["winner_dendrite"])
    print("Latency (soma processing)   :", result["latency"])
    print("Absolute spike time         :", result["abs_spike_time"])

    # 4) Interne Größen für Diagnose holen
    diag = analyze_encoder(model, x)

    # 5) Plots

    # (a) Signal + bestes Fenster
    plot_signal_with_best_window(x_np, result["pattern_index"], W)

    # (b) Winner-Strom + Latenzen über Fenster
    plot_I_and_latencies(diag["I"], diag["latencies"])

    # (c) Soma-Traces für ein paar Fenster (z.B. Gewinner, Nachbarn)
    best = result["pattern_index"]
    N_windows = diag["I"].shape[0]
    cand = sorted(set([
        best,
        max(0, best - 5),
        min(N_windows - 1, best + 5)
    ]))
    plot_soma_traces_for_windows(model, diag["I"], cand)

    # (d) Dendritische Aktivierung für Gewinnerfenster
    plot_dendritic_activations_for_window(diag, best)
