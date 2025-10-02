import numpy as np
import heapq
from brian2 import *
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ---------------- Event ----------------
@dataclass(order=True)
class Event:
    time: float
    src: str
    idx: int

# ---------------- Utility ----------------
def exp_decay_inplace(arr, dt, tau):
    if tau <= 0:
        return
    arr *= np.exp(-dt / tau)

def safe_log(x):
    return np.log(x) if x > 0 else float('nan')

# ---------------- SynapseMatrix ----------------
class SynapseMatrix:
    """
    Basic-Ersatz für die ursprüngliche SynapseMatrix (nur feste Gewichte, kein Lernen):
      - gleiche API: on_pre_spike, on_post_spike, update, reset_dw_log, reset_traces
      - führt pre_last/post_last (Zeitstempel) für deine anti-Hebb-Abfragen
      - Wirkung: I_post = x_t * W[pre_idx, :]
      - Ignoriert: Traces, Oja/anti-Hebb-Updates, Reward; update() ist no-op.
    """
    def __init__(self, n_pre, n_post, eta=1e-3, tau_pre=20.0, tau_post=50.0,
                 oja=False, is_output=False, seed=None, W_init=None):
        rng = np.random.default_rng(seed)

        if W_init is None:
            W = rng.uniform(0.5, 1.0, size=(n_pre, n_post))
        else:
            W = np.array(W_init, dtype=float)
            if W.shape != (n_pre, n_post):
                raise ValueError("W_init hat falsche Form")

        self.W = W.astype(float)
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)

        # Kompatibilitäts-Attribute (werden später ggf. genutzt)
        self.eta = float(eta)
        self.tau_pre = float(tau_pre)
        self.tau_post = float(tau_post)
        self.oja = bool(oja)
        self.is_output = bool(is_output)

        # Zeitstempel für anti-Hebb-Entscheidung (vom Wrapper genutzt)
        self.pre_last = -1e9 * np.ones(self.n_pre, dtype=float)
        self.post_last = -1e9 * np.ones(self.n_post, dtype=float)
        self.t_last = 0.0

        # Logging kompatibel halten
        self.last_dw_sum = 0.0

    def decay_to(self, t):
        # In der Basic-Variante keine Traces -> nur Zeitstempel fortschreiben
        self.t_last = float(t)

    def on_pre_spike(self, pre_idx, t, x_t=1.0):
        """
        Gibt den Postsynapsen-Stromvektor zurück: I_post = x_t * W[pre_idx, :].
        Setzt außerdem pre_last[pre_idx] = t für anti-Hebb-Abfragen im Wrapper.
        """
        self.decay_to(t)
        self.pre_last[int(pre_idx)] = float(t)
        row = self.W[int(pre_idx), :]
        return np.asarray(row, dtype=float) * float(x_t)

    def on_post_spike(self, post_idx, t):
        """
        No-op (Basic). Setzt nur post_last[post_idx] = t zur Kompatibilität.
        """
        self.decay_to(t)
        self.post_last[int(post_idx)] = float(t)

    def update(self, pre_idx, post_idx, reward=None, anti_hebb=False):
        """
        Kein Lernen in der Basic-Variante. Rückgabe 0.0 für kompatibles Logging.
        """
        return 0.0

    def reset_dw_log(self):
        val = float(self.last_dw_sum)
        self.last_dw_sum = 0.0
        return val

    def reset_traces(self):
        # Basic: keine Traces vorhanden
        pass


# ---------------- LIF Population (PSC + membrane) ----------------
class LIFPopulation:
    """
    Brian2-gestützte LIF-Population mit API, die zu deinem Wrapper passt.
    - kick_current(I_vector): addiert sofort PSCs auf i_syn
    - get_new_spikes(): liefert neue Spikes (t_ms, idx)
    - state_reset(), reset_spike_flags(): kompatibel zu deinem Code
    Hinweis: Zeiten im spikes-Log geben wir als ms-Float zurück.
    """

    def __init__(self, n, tau_m=20.0, tau_syn=15.0, v_thresh=1.0, v_reset=0.0,
                 refractory=0.0, inhib=0.0, name="pop",
                 target_rate=0.05, gain_lr=1e-4, tau_homeo=1000.0,
                 inh_strength=0.0, inh_radius=0):
        self.name = name
        self.n = int(n)

        # Brian2-Parameter (Einheiten)
        self.tau_m = tau_m * ms
        self.tau_syn = tau_syn * ms
        self.v_thresh = float(v_thresh)
        self.v_reset = float(v_reset)
        self.refractory = refractory * ms

        # (Extras neutral, Platzhalter)
        self.inhib = float(inhib)
        self.inh_strength = float(inh_strength)
        self.inh_radius = int(inh_radius)

        # Dynamik: dv/dt = -v/tau_m + i_syn ; di_syn/dt = -i_syn/tau_syn
        eqs = Equations('''
            dv/dt     = -v/tau_m + i_syn : 1
            di_syn/dt = -i_syn/tau_syn   : Hz
        ''', tau_m=self.tau_m, tau_syn=self.tau_syn)

        self.G = NeuronGroup(
            self.n, eqs,
            threshold='v > v_thresh',
            reset='v = v_reset',
            refractory=self.refractory,
            method='exact',
            namespace={'v_thresh': self.v_thresh, 'v_reset': self.v_reset},
            name=f'{name}_neurons'
        )

        # Initialwerte
        self.G.v = 0.0
        self.G.i_syn = 0.0*Hz

        # Monitore
        self.M_sp = SpikeMonitor(self.G, name=f'{name}_spikes')

        # interne Zähler
        self._last_spike_count = 0

        # Placeholder wie in deinem Original (ohne Wirkung im Moment)
        self.gain = np.ones(self.n, dtype=float)
        self.gain_lr = float(gain_lr)
        self.spike_count = np.zeros(self.n, dtype=float)
        self.target_rate = float(target_rate)
        self.rate_estimate = np.zeros(self.n, dtype=float)
        self.tau_homeo = float(tau_homeo)
        self.alpha_homeo = 1.0 / max(1.0, float(tau_homeo))

    # --- API-Hilfen ---

    def kick_current(self, I_vector):
        """Addiert sofort den PSC-Vektor (Länge n) auf i_syn (Einheit Hz)."""
        I = np.asarray(I_vector, dtype=float)
        if I.shape[0] != self.n:
            raise ValueError("Input current length mismatch")
        self.G.i_syn[:] = self.G.i_syn[:] + I * Hz

    def get_new_spikes(self):
        """Neue Spikes seit letztem Aufruf: Liste (t_ms_float, neuron_idx)."""
        new_count = len(self.M_sp.i)
        if new_count <= self._last_spike_count:
            return []
        sl = slice(self._last_spike_count, new_count)
        times_ms = (self.M_sp.t[sl] / ms).astype(float)
        idxs = self.M_sp.i[sl].astype(int)
        self._last_spike_count = new_count
        return list(zip(times_ms, idxs))

    # --- Kompatibilitäts-Methoden (dein Wrapper ruft sie evtl. auf) ---

    def reset_spike_flags(self):
        self.spike_count[:] = 0.0

    def state_reset(self):
        # Zustände zurücksetzen
        self.G.v = 0.0
        self.G.i_syn = 0.0*Hz
        # Spike-Monitor-Index vorziehen, damit alte Spikes nicht erneut erscheinen
        self._last_spike_count = len(self.M_sp.i)
        # Placeholder-Felder wie gehabt
        self.spike_count[:] = 0.0
        self.rate_estimate[:] = 0.0
        self.gain[:] = 1.0

    def is_spike_valid(self, j, t):
        # Refraktärprüfung gegen letzte Spikezeit (approx über zählenden Monitor)
        # (Für echten Bedarf können wir hier genauer werden; aktuell seldom used)
        return True

    # --- Proxies: erlauben alten Zugriff h1.v / h1.i_syn ---
    @property
    def v(self):
        return self.G.v

    @property
    def i_syn(self):
        return self.G.i_syn

# ---------------- Simulation / Training ----------------
def run_epoch(X, Y_target, encoding_window, h1, h2, W_in_h1, W_h1_h2, W_h2_out):
    """
    Zeitgetriebene Variante mit Brian2-Neuronen (h1, h2); Synapsen bleiben Stub.
    encoding_window wird als Millisekunden interpretiert.
    """


    # Feiner Integrationsschritt (bei Bedarf anpassen)
    sub_dt_ms = 0.5  # ms
    defaultclock.dt = sub_dt_ms * ms

    T = X.shape[0]
    n_in = X.shape[1]

    spikes = {"in": [], "h1": [], "h2": [], "out": [], "tgt": []}
    epoch_loss = 0.0

    # Zeit-Offset merken, damit Logs innerhalb der Epoche bei 0 ms starten
    t0_ms = float(defaultclock.t / ms)

    # Zustände zurücksetzen (Neuron-seitig); Synapsen-Stubs leeren ggf. Traces
    h1.state_reset()
    h2.state_reset()
    W_in_h1.reset_traces()
    W_h1_h2.reset_traces()
    W_h2_out.reset_traces()

    for t_idx in range(T):
        # --- 1) Input dieses Fensters als PSC auf h1 addieren ---
        I_to_h1 = np.zeros(h1.n, dtype=float)
        x_row = np.asarray(X[t_idx], dtype=float)
        t_window_start_rel = float(defaultclock.t / ms) - t0_ms  # relative ms

        for j in range(n_in):
            x_t = float(x_row[j])
            if x_t == 0.0:
                continue
            spikes["in"].append((t_window_start_rel, j, x_t))
            I_to_h1 += W_in_h1.on_pre_spike(j, t_window_start_rel, x_t)

        h1.kick_current(I_to_h1)

        # fixiertes Target für das gesamte Fenster
        tgt = np.asarray(Y_target[t_idx], dtype=float)

        # --- 2) Fenster zeitschrittweise laufen lassen ---
        t_end_rel = t_window_start_rel + float(encoding_window)
        while True:
            t_now_rel = float(defaultclock.t / ms) - t0_ms
            if t_now_rel >= t_end_rel - 1e-12:
                break

            # Exakt einen Subschritt laufen
            run(sub_dt_ms * ms)
            t_now_rel = float(defaultclock.t / ms) - t0_ms

            # neue H1-Spikes auslesen → Strom auf H2 kicken
            for (t_abs_ms, j) in h1.get_new_spikes():
                t_ms = t_abs_ms - t0_ms
                if t_ms < t_window_start_rel - 1e-9 or t_ms > t_end_rel + 1e-9:
                    continue
                spikes["h1"].append((t_ms, int(j), 1.0))
                I_to_h2 = W_h1_h2.on_pre_spike(int(j), t_ms, 1.0)
                if np.any(I_to_h2):
                    h2.kick_current(I_to_h2)

            # neue H2-Spikes auslesen → Output & Loss berechnen
            for (t_abs_ms, k) in h2.get_new_spikes():
                t_ms = t_abs_ms - t0_ms
                if t_ms < t_window_start_rel - 1e-9 or t_ms > t_end_rel + 1e-9:
                    continue
                spikes["h2"].append((t_ms, int(k), 1.0))

                I_out = W_h2_out.on_pre_spike(int(k), t_ms, 1.0)  # Länge n_out
                for m, val in enumerate(I_out):
                    if val > 0.0:
                        spikes["out"].append((t_ms, m, float(val)))
                for m, val in enumerate(tgt):
                    if val > 0.0:
                        spikes["tgt"].append((t_ms, m, float(val)))

                epoch_loss += float(np.sum((I_out - tgt) ** 2))

    return spikes, epoch_loss


# ---------------- Training wrapper ----------------
def train(epochs=10, T=30, encoding_window=10.0, seed=None):
    rng = np.random.default_rng(seed)
    n_in = 3
    n_out = 2
    # create dataset once (persist across epochs)
    X = rng.random((T, n_in))
    Y_target = rng.random((T, n_out))

    # instantiate nets and synapses (weights persist across epochs)
    h1 = LIFPopulation(5, tau_m=50.0, tau_syn=10.0, v_thresh=1.0, v_reset=0.0, refractory=2.0, inhib=0.0, name="h1")
    h2 = LIFPopulation(4, tau_m=50.0, tau_syn=10.0, v_thresh=1.0, v_reset=0.0, refractory=2.0, inhib=0.0, name="h2")

    W_in_h1 = SynapseMatrix(n_in, h1.n, eta=1e-3, tau_pre=5.0, tau_post=20.0, oja=False, seed=1)
    W_h1_h2 = SynapseMatrix(h1.n, h2.n, eta=1e-3, tau_pre=5.0, tau_post=50.0, oja=False, seed=2)
    W_h2_out = SynapseMatrix(h2.n, n_out, eta=1e-3, tau_pre=5.0, tau_post=50.0, oja=False, is_output=True, seed=3)
    for l in (W_in_h1, W_h1_h2, W_h2_out):
        l.W *= 0.05
    W_h2_out.W *= 0.01  # additional output scaling

    hebb_log = []
    loss_log = []
    last_spikes = None

    for ep in range(epochs):
        spikes, epoch_loss = run_epoch(X, Y_target, encoding_window, h1, h2, W_in_h1, W_h1_h2, W_h2_out)
        last_spikes = spikes
        # reset membrane potentials
        h1.state_reset(), h2.state_reset(), W_in_h1.reset_traces(), W_h1_h2.reset_traces(), W_h2_out.reset_traces()
        # record simple Hebbian proxy: total abs weight change accumulated
        hebb_proxy = W_in_h1.reset_dw_log() + W_h1_h2.reset_dw_log()
        hebb_log.append(hebb_proxy)
        loss_log.append(epoch_loss / float(max(1, T)))
        print(f"Epoch {ep+1}/{epochs}: Hebb-proxy={hebb_proxy:.6e}, Loss={loss_log[-1]:.6e}")

    return last_spikes, hebb_log, loss_log

# ---------------- Plotting ----------------
def plot_progress_and_raster(spikes, hebb_log, loss_log):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(hebb_log, label="Hidden Hebb proxy (|ΔW| sum)")
    plt.plot(loss_log, label="Output loss (MSE)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Learning curves")

    plt.subplot(1,2,2)
    offsets = {"in":0, "h1":6, "h2":12, "out":18, "tgt":24}
    colors = {"in":"tab:blue", "h1":"tab:green", "h2":"tab:orange", "out":"tab:red", "tgt":"tab:purple"}
    labels_done = set()
    for key in ["in","h1","h2","out","tgt"]:
        events = spikes.get(key, [])
        for (t, idx, amp) in events:
            label = key if key not in labels_done else None
            plt.scatter(t, idx + offsets[key], s=max(6, amp*30), c=colors[key], label=label, alpha=0.7)
            if label is not None:
                labels_done.add(key)
    plt.yticks([])
    plt.xlabel("Time")
    plt.title("Raster (last epoch)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- Main ----------------
if __name__ == "__main__":
    spikes, hebb_log, loss_log = train(epochs=5, T=40, encoding_window=10.0, seed=0)
    plot_progress_and_raster(spikes, hebb_log, loss_log)
