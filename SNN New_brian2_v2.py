import numpy as np
import heapq
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
import numpy as np

class LIFPopulation:
    """
    Adapter-Implementierung der LIF-Population im eventgetriebenen Stil,
    API-kompatibel zu deinem bestehenden run_epoch.

    Hinweis: Diese Version bildet exakt das Verhalten des bisherigen
    eventgetriebenen LIF nach (Analytik für v/i_syn, Spike-Zeit-Vorhersage),
    damit dein Wrapper unverändert weiterläuft. Im nächsten Schritt können
    wir die Interna schrittweise auf Brian2 umstellen, ohne die äußere API
    zu verändern.
    """
    def __init__(self, n, tau_m=20.0, tau_syn=15.0, v_thresh=1.0, v_reset=0.0,
                 refractory=0.0, inhib=0.0, name="pop",
                 target_rate=0.05, gain_lr=1e-4, tau_homeo=1000.0,
                 inh_strength=0.0, inh_radius=0):
        self.name = name
        self.n = int(n)

        # Zeitkonstanten & Schwellen
        self.tau_m = float(tau_m)
        self.tau_syn = float(tau_syn)
        self.v_thresh = float(v_thresh)
        self.v_reset = float(v_reset)
        self.refractory = float(refractory)

        # (Extras zunächst neutral)
        self.inhib = float(inhib)
        self.inh_strength = float(inh_strength)
        self.inh_radius = int(inh_radius)

        # Zustände
        self.v = np.zeros(self.n, dtype=float)
        self.i_syn = np.zeros(self.n, dtype=float)
        self.t_last_spike = -1e9 * np.ones(self.n, dtype=float)

        # Homeostase-Placeholder (derzeit neutral)
        self.gain = np.ones(self.n, dtype=float)
        self.gain_lr = float(gain_lr)
        self.spike_count = np.zeros(self.n, dtype=float)
        self.target_rate = float(target_rate)
        self.rate_estimate = np.zeros(self.n, dtype=float)
        self.tau_homeo = float(tau_homeo)
        self.alpha_homeo = 1.0 / max(1.0, float(tau_homeo))

        # Zeitbuchhaltung pro Neuron
        self.t_last_update = np.zeros(self.n, dtype=float)

    # -------- Zustände zu Zeit t abklingen lassen --------
    def decay_to(self, t):
        dt = t - self.t_last_update
        mask = dt > 0
        if not np.any(mask):
            return
        dt_mask = dt[mask]
        self.v[mask] *= np.exp(-dt_mask / self.tau_m)
        self.i_syn[mask] *= np.exp(-dt_mask / self.tau_syn)
        self.rate_estimate[mask] *= np.exp(-dt_mask / self.tau_homeo)
        self.t_last_update[mask] = t

    # -------- analytische Spikezeit-Vorhersage --------
    def compute_next_spike(self, j, I_drive, t_now, V_before):
        t_earliest = max(t_now, self.t_last_spike[j] + self.refractory)
        V0 = V_before
        Vth = self.v_thresh
        V_inf = I_drive * self.tau_m
        if V_inf <= Vth:
            return None
        if V0 >= Vth and t_now >= t_earliest:
            return t_now
        denom = (V0 - V_inf)
        num = (Vth - V_inf)
        if denom == 0.0:
            return None
        ratio = num / denom
        if ratio <= 0.0:
            return None
        # sichere Log
        val = np.log(ratio) if ratio > 0 else np.nan
        if not np.isfinite(val):
            return None
        dt_cross = - self.tau_m * val
        if not np.isfinite(dt_cross) or dt_cross < 0.0:
            return None
        t_cross = t_now + dt_cross
        if t_cross < t_earliest:
            return None
        return t_cross

    # -------- eingehenden Stromvektor verarbeiten --------
    def receive_current(self, I_vector, t_now):
        self.decay_to(t_now)
        I = np.asarray(I_vector, dtype=float)
        if I.shape[0] != self.n:
            raise ValueError("Input current length mismatch")
        events = []
        for j in range(self.n):
            if (t_now - self.t_last_spike[j]) < self.refractory:
                continue
            V_before = self.v[j]
            self.i_syn[j] += I[j] * self.gain[j]
            t_cross = self.compute_next_spike(j, self.i_syn[j], t_now, V_before)
            if t_cross is not None:
                events.append((float(t_cross), int(j)))
        return events

    # -------- Spike registrieren --------
    def register_spike(self, j, t):
        self.decay_to(t)
        self.v[j] = self.v_reset
        self.t_last_spike[j] = t
        self.spike_count[j] += 1
        # Laterale Inhibition (derzeit neutral, nur falls gesetzt)
        if self.inh_strength != 0.0 and self.inh_radius > 0:
            for offset in range(-self.inh_radius, self.inh_radius + 1):
                k = j + offset
                if 0 <= k < self.n and k != j:
                    self.v[k] -= self.inh_strength
        # Homeostase (Placeholder)
        self.rate_estimate[j] += self.alpha_homeo
        self.gain += self.gain_lr * (self.target_rate - self.rate_estimate)
        self.gain = np.clip(self.gain, 0.1, 10.0)

    def is_spike_valid(self, j, t):
        return (t - self.t_last_spike[j]) >= self.refractory

    # -------- Hauskeeping --------
    def reset_spike_flags(self):
        self.spike_count[:] = 0.0

    def state_reset(self):
        self.v[:] = 0.0
        self.i_syn[:] = 0.0
        self.t_last_spike[:] = -1e9
        self.t_last_update[:] = 0.0
        self.spike_count[:] = 0.0
        self.rate_estimate[:] = 0.0
        self.gain[:] = 1.0

    # ---------- decay states to time t ----------
    def decay_to(self, t):
        """
        Exponentially decay v, i_syn, and rate_estimate from each neuron's last update time
        to time t. This handles asynchronous event-driven progression.
        """
        dt = t - self.t_last_update
        mask = dt > 0
        if not np.any(mask):
            return
        dt_mask = dt[mask]
        self.v[mask] *= np.exp(-dt_mask / self.tau_m)
        self.i_syn[mask] *= np.exp(-dt_mask / self.tau_syn)
        # decay rate estimate according to tau_homeo
        self.rate_estimate[mask] *= np.exp(-dt_mask / self.tau_homeo)
        # update last update times
        self.t_last_update[mask] = t

    # ---------- analytic spike time prediction ----------
    def compute_next_spike(self, j, I_drive, t_now, V_before):
        """
        Predict next crossing time for neuron j assuming I_drive (PSC) stays constant.
        Solves V(t) = V_inf + (V0 - V_inf)*exp(-(t-t_now)/tau_m) = V_thresh.
        Returns absolute t_cross or None.
        """
        t_earliest = max(t_now, self.t_last_spike[j] + self.refractory)

        V0 = V_before
        Vth = self.v_thresh
        V_inf = I_drive * self.tau_m  # steady-state voltage for constant current

        # if steady-state below threshold, never cross
        if V_inf <= Vth:
            return None

        # if already above threshold and not in refractory -> immediate spike
        if V0 >= Vth and t_now >= t_earliest:
            return t_now

        denom = (V0 - V_inf)
        num = (Vth - V_inf)
        if denom == 0.0:
            return None
        ratio = num / denom
        if ratio <= 0.0:
            return None

        dt_cross = - self.tau_m * safe_log(ratio)
        if not np.isfinite(dt_cross) or dt_cross < 0.0:
            return None

        t_cross = t_now + dt_cross
        if t_cross < t_earliest:
            return None

        return t_cross

    # ---------- receive incoming vector current (from synapses) ----------
    def receive_current(self, I_vector, t_now):
        """
        I_vector: numpy array length n with instantaneous PSC additions (e.g. from on_pre_spike).
        We:
          - decay internal states to t_now
          - apply divisive inhibition across the vector
          - add scaled currents to i_syn (PSC)
          - compute candidate crossing times for neurons not refractory
        Returns list of (t_cross, neuron_idx)
        """
        self.decay_to(t_now)
        I = np.asarray(I_vector, dtype=float)
        if I.shape[0] != self.n:
            raise ValueError("Input current length mismatch")

        # optional divisive inhibition on the incoming vector (global competition)
        total = np.sum(I)
        if total > 0.0 and self.inhib > 0.0:
            I = I / (1.0 + self.inhib * total)

        events = []
        for j in range(self.n):
            if (t_now - self.t_last_spike[j]) < self.refractory:
                continue
            V_before = self.v[j]
            # accumulate PSC (gain scales synaptic efficacy)
            self.i_syn[j] += I[j] * self.gain[j]
            # predict crossing using current i_syn (assumed constant forward)
            t_cross = self.compute_next_spike(j, self.i_syn[j], t_now, V_before)
            if t_cross is not None:
                events.append((t_cross, j))
        return events

    # ---------- spike registration (called when a spike event occurs) ----------
    def register_spike(self, j, t):
        """
        Reset membrane V, record spike time, apply local lateral inhibition,
        update rate estimate and homeostatic gain for the spiking neuron.
        """
        # make sure states are decayed to t before applying instantaneous updates
        self.decay_to(t)

        # reset membrane and bookkeeping
        self.v[j] = self.v_reset
        self.t_last_spike[j] = t
        self.spike_count[j] += 1

        # LOCAL lateral inhibition: only affect neighbors within inh_radius
        for offset in range(-self.inh_radius, self.inh_radius + 1):
            k = j + offset
            if 0 <= k < self.n and k != j:
                # subtract from membrane to transiently delay neighbors
                self.v[k] -= self.inh_strength

        # HOMEOSTASIS / rate estimate update:
        # (we decayed rate_estimate in decay_to above so it's up-to-date)
        # add a small bump for this spike (alpha_homeo chosen ~ 1/tau_homeo)
        self.rate_estimate[j] += self.alpha_homeo

        # adjust gain (can be vectorized; only j changed but we keep simple)
        # gain moves slowly toward keeping rate_estimate ~ target_rate
        self.gain += self.gain_lr * (self.target_rate - self.rate_estimate)
        # clamp gains to prevent runaway
        self.gain = np.clip(self.gain, 0.1, 10.0)

    def is_spike_valid(self, j, t):
        return (t - self.t_last_spike[j]) >= self.refractory

    # ---------- housekeeping ----------
    def reset_spike_flags(self):
        self.spike_count[:] = 0.0

    def state_reset(self):
        self.v[:] = 0.0
        self.i_syn[:] = 0.0
        self.t_last_spike[:] = -1e9
        self.t_last_update[:] = 0.0
        self.spike_count[:] = 0.0
        self.rate_estimate[:] = 0.0
        self.gain[:] = 1.0

# ---------------- Simulation / Training ----------------
def run_epoch(X, Y_target, encoding_window, h1, h2, W_in_h1, W_h1_h2, W_h2_out):
    """
    Single epoch over dataset X, Y_target (shape T x n_in, T x n_out).
    Returns spikes dict and epoch_loss (sum of sample losses).
    """
    T = X.shape[0]
    n_in = X.shape[1]
    evq = []
    spikes = {"in": [], "h1": [], "h2": [], "out": [], "tgt": []}
    epoch_loss = 0.0

    # preload input events (one event per input neuron per sample time)
    for t_idx in range(T):
        t_ev = t_idx * encoding_window
        for j in range(n_in):
            heapq.heappush(evq, Event(time=float(t_ev), src="in", idx=int(j)))

    I_to_h1_reg = []
    # process event queue
    while evq:
        ev = heapq.heappop(evq)
        t = ev.time
        sample_idx = min(int(t // encoding_window), T - 1)

        # INPUT pre-spike
        if ev.src == "in":
            x_t = float(X[sample_idx, ev.idx])
            spikes["in"].append((t, ev.idx, x_t))
            # update presyn trace and get postsyn currents
            I_to_h1 = W_in_h1.on_pre_spike(ev.idx, t, x_t)    # length n_h1
            I_to_h1_reg.append(I_to_h1)
            # deliver currents to h1 (accumulate PSCs) and schedule predicted spikes
            new_events = h1.receive_current(I_to_h1, t)
            for (t_cross, j) in new_events:
                heapq.heappush(evq, Event(time=float(t_cross), src="h1", idx=int(j)))

        # H1 post-spike
        elif ev.src == "h1":
            j = ev.idx
            # TODO: is this needed?
            if not h1.is_spike_valid(j, t):
                continue
            # register spike
            h1.register_spike(j, t)
            spikes["h1"].append((t, j, 1.0))
            # learning: update pre/post traces at synapse
            W_in_h1.on_post_spike(j, t)
            # update input->h1 weights: for each pre_idx decide anti_hebb by comparing last times
            for pre_idx in range(W_in_h1.W.shape[0]):
                pre_t = W_in_h1.pre_last[pre_idx]
                post_t = W_in_h1.post_last[j]
                anti = (pre_t > post_t)   # if pre happened after post => anti-Hebbian
                W_in_h1.update(pre_idx, j, reward=None, anti_hebb=anti)
            # propagate to h2
            I_to_h2 = W_h1_h2.on_pre_spike(j, t, 1.0)   # amplitude 1.0 from hidden spike
            new_events = h2.receive_current(I_to_h2, t)
            for (t_cross, k) in new_events:
                heapq.heappush(evq, Event(time=float(t_cross), src="h2", idx=int(k)))

        # H2 post-spike -> output & learning
        elif ev.src == "h2":
            k = ev.idx
            # TODO: needed?
            if not h2.is_spike_valid(k, t):
                continue
            h2.register_spike(k, t)
            spikes["h2"].append((t, k, 1.0))
            # learning h1->h2
            W_h1_h2.on_post_spike(k, t)
            for pre_idx in range(W_h1_h2.W.shape[0]):
                pre_t = W_h1_h2.pre_last[pre_idx]
                post_t = W_h1_h2.post_last[k]
                anti = (pre_t > post_t)
                W_h1_h2.update(pre_idx, k, reward=None, anti_hebb=anti)

            # forward to output
            I_out = W_h2_out.on_pre_spike(k, t, 1.0)  # length n_out
            for m, val in enumerate(I_out):
                if val > 0:
                    spikes["out"].append((t, m, float(val)))
            # target (for this sample)
            tgt = np.asarray(Y_target[sample_idx], dtype=float)
            for m, val in enumerate(tgt):
                if val > 0:
                    spikes["tgt"].append((t, m, float(val)))

            # learning at output: for each post-output neuron that has current
            reward = float(np.exp(-np.sum((I_out - tgt) ** 2)))
            for post_idx in range(W_h2_out.W.shape[1]):
                if I_out[post_idx] <= 0.0:
                    continue
                W_h2_out.on_post_spike(post_idx, t)
                for pre_idx in range(W_h2_out.W.shape[0]):
                    # anti-hebb decision same way (optional)
                    pre_t = W_h2_out.pre_last[pre_idx]
                    post_t = W_h2_out.post_last[post_idx]
                    anti = (pre_t > post_t)
                    W_h2_out.update(pre_idx, post_idx, reward=reward, anti_hebb=anti)

            # supervised loss accumulation (MSE)
            epoch_loss += np.sum((I_out - tgt) ** 2)

    # print(I_to_h1_reg)
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
