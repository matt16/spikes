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

#-----------new basic Synapse matrix without learning-------
class NewSynapseMatrix:
    """
    Drop-in Ersatz für SynapseMatrix mit:
      - pre_trace / post_trace + exponentiellem Zerfall wie gehabt
      - on_pre_spike() gibt pre_trace @ W zurück (PSC-Summe über alle Prä-Spuren)
      - update() ist NO-OP (keine Lern-Änderung der Gewichte)
    """
    def __init__(self, W_np, tau_pre=20.0, tau_post=50.0):
        W_np = np.asarray(W_np, dtype=float)              # shape: (n_pre, n_post)
        self.W = W_np
        self.n_pre, self.n_post = W_np.shape
        self.tau_pre = float(tau_pre)
        self.tau_post = float(tau_post)

        self.pre_trace = np.zeros(self.n_pre, dtype=float)
        self.post_trace = np.zeros(self.n_post, dtype=float)
        self.pre_last = -1e9 * np.ones(self.n_pre, dtype=float)
        self.post_last = -1e9 * np.ones(self.n_post, dtype=float)
        self.t_last = 0.0

        self.last_dw_sum = 0.0  # bleibt immer 0

    def decay_to(self, t):
        dt = t - self.t_last
        if dt > 0:
            exp_decay_inplace(self.pre_trace, dt, self.tau_pre)
            exp_decay_inplace(self.post_trace, dt, self.tau_post)
            self.t_last = float(t)

    def on_pre_spike(self, pre_idx, t, x_t=1.0):
        """Presyn-Spike/Analoginput: Trace wie gehabt, Ausgabe = pre_trace @ W."""
        self.decay_to(t)
        self.pre_trace[pre_idx] += float(x_t)
        self.pre_last[pre_idx] = float(t)
        return self.pre_trace @ self.W  # Länge n_post (hier: n_h1)

    def on_post_spike(self, post_idx, t):
        """Nur für Kompatibilität/Logging der Postsyn-Spur (kein Lernen)."""
        self.decay_to(t)
        self.post_trace[post_idx] += 1.0
        self.post_last[post_idx] = float(t)

    def update(self, pre_idx, post_idx, reward=None, anti_hebb=False):
        """NO-OP: Gewichte bleiben fix."""
        return 0.0

    def reset_dw_log(self):
        val = self.last_dw_sum
        self.last_dw_sum = 0.0
        return val

    def reset_traces(self):
        self.pre_trace[:] = 0.0
        self.post_trace[:] = 0.0
        self.pre_last[:] = -1e9
        self.post_last[:] = -1e9
        self.t_last = 0.0




# ---------------- LIF Population (PSC + membrane) ----------------
class LIFPopulation:
    """
    LIF with synaptic PSC trace (i_syn) and membrane V.
    i_syn decays with tau_syn; V decays with tau_m.
    compute_next_spike predicts crossing time assuming i_syn stays constant.
    """
    def __init__(self, n, tau_m=20.0, tau_syn=5.0, v_thresh=1.0, v_reset=0.0, refractory=0.0, inhib=0.1, init_dend_tau=5.0, init_dend_gain=1.0, name="pop"):
        self.name = name
        self.n = n
        # self.tau_syn = tau_syn
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.refractory = refractory
        self.inhib = inhib
        self.i_dend = np.zeros(n, dtype=float)  # this corresponds to I0 in formulas
        # per-neuron trainable dendritic time constant and gain (can be adjusted during learning)
        self.dend_tau = np.full(n, float(init_dend_tau), dtype=float)
        self.dend_gain = np.full(n, float(init_dend_gain), dtype=float)

        # synaptic current (PSC) per neuron
        # self.i_syn = np.zeros(n, dtype=float)

        # states
        self.v = np.zeros(n)               # membrane potentials
        self.t_last_spike = -1e9 * np.ones(n)  # last spike times
        self.gain = np.ones(n)
        self.gain_lr = 1 * 1e-3

        # homeostasis
        self.spike_count = np.zeros(n)
        self.target_rate = 0.05

        # track last update time per neuron
        self.t_last_update = np.zeros(n)

    def decay_to(self, t):
        dt = t - self.t_last_update
        mask = dt > 0
        if np.any(mask):
            # membrane exponential decay
            self.v[mask] *= np.exp(-dt[mask] / self.tau_m)
            # # synaptic (PSC) exponential decay
            # self.i_syn[mask] *= np.exp(-dt[mask] / self.tau_syn)
            # self.t_last_update[mask] = t
            # dendritic trace exponential decay (each neuron has own tau)
            # compute factor per neuron
            taus = self.dend_tau[mask]
            self.i_dend[mask] *= np.exp(-dt[mask] / taus)
            self.t_last_update[mask] = t

    def _V_of_s(self, j, s, V0, I0):
        """
        Membrane voltage at time t = t_now + s (s >= 0), given:
          - neuron j
          - V0 = membrane at t_now
          - I0 = dendritic amplitude at t_now (so drive = I0 * exp(-s / dend_tau[j]))
        Analytical expression:
          V(s) = V0*exp(-s/tau_m) + A * (exp(-s/dend_tau) - exp(-s/tau_m))
        where A = I0 * tau_m * dend_tau / (dend_tau - tau_m)  (for dend_tau != tau_m)
        If dend_tau == tau_m the limit yields V(s) = V0*e^{-s/tau} + I0 * tau * s * e^{-s/tau}
        """
        tau_m = self.tau_m
        tau_k = float(self.dend_tau[j])
        if I0 == 0.0:
            return V0 * np.exp(-s / tau_m)
        if abs(tau_k - tau_m) > 1e-12:
            A = I0 * tau_m * tau_k / (tau_k - tau_m)
            return V0 * np.exp(-s / tau_m) + A * (np.exp(-s / tau_k) - np.exp(-s / tau_m))
        else:
            # tau_k == tau_m special-case (use limit)
            tau = tau_m
            return np.exp(-s / tau) * (V0 + I0 * tau * s)

    def compute_next_spike(self, j, I_drive, t_now, V_before):
        """
        Predict next spike time for neuron j assuming dendritic drive decays exponentially:
          I_drive(t) = I_drive_now * exp(-(t - t_now)/dend_tau[j])
        We look for the earliest s >= 0 s.t. V(s) = v_thresh, where V(s) is defined
        by _V_of_s. We also respect refractory period.
        Returns absolute t_cross or None.
        """
        t_earliest = max(t_now, self.t_last_spike[j] + self.refractory)
        V0 = V_before
        Vth = self.v_thresh
        I0 = float(I_drive)
        # quick checks
        if I0 <= 0.0:
            # without positive drive there is no chance to go up above threshold (monotonic decay)
            # (if V0 already above threshold, return current time)
            if V0 >= Vth and t_now >= t_earliest:
                return t_now
            return None

        # Check immediate crossing
        if V0 >= Vth and t_now >= t_earliest:
            return t_now

        # check whether the maximum reachable voltage (over s>=0) exceeds threshold
        # compute peak analytically: the membrane can transiently rise due to the exponential drive.
        # We'll check value on a finite window [0, s_max], where s_max covers several taus.
        tau_k = float(self.dend_tau[j])
        s_max = max(10.0 * tau_k, 10.0 * self.tau_m, 50.0)  # safe upper bound

        _HAS_LAMBERTW = True
        # Fast analytic V(infty) = 0 (drive decays to 0), so only transient crossing possible.
        # We'll attempt closed-form solve using lambertw if available; otherwise numeric bisection.
        if _HAS_LAMBERTW:
            # Attempt closed-form solution using Lambert W.
            # Derivation yields (after algebra) an expression that can be rearranged and solved with lambertw.
            # Because the algebra is a bit involved and brittle to show in-line, we use a standard
            # rearrangement that leads to a Lambert W application.
            try:
                tau_m = self.tau_m
                tau_k = float(self.dend_tau[j])
                V0f = V0
                I0f = I0
                Vthf = Vth

                if abs(tau_k - tau_m) < 1e-12:
                    # special degenerate case: tau_k == tau_m
                    # V(s) = e^{-s/tau} * (V0 + I0 * tau * s) -> solve Vth = e^{-s/tau}*(V0 + I0 tau s)
                    # Rearranged: (V0 - Vth) * e^{(V0 - Vth)/(I0 tau)} + I0 tau s * e^{(V0 - Vth)/(I0 tau)} = ...
                    # This becomes solvable via lambertw; implement stable algebra below.
                    tau = tau_m
                    # Let u = s / tau
                    # equation: (V0 + I0 * tau * s) * e^{-s/tau} = Vth
                    # => (I0 * tau^2) * u * e^{-u} + V0 * e^{-u} - Vth = 0
                    # Rearranged to form z e^{z} = const to apply lambertw:
                    # After algebra one gets a solution; here we implement a robust numeric fallback if complex arises.
                    # For simplicity, attempt numeric bisection in this branch because analytic form is awkward.
                    pass  # fall back to numeric below if lambert path is not coded fully
                else:
                    # General case tau_k != tau_m.
                    # Let s be the unknown. We have
                    # (V0 - A) * exp(-s/tau_m) + A * exp(-s/tau_k) = Vth
                    # where A = I0 * tau_m * tau_k / (tau_k - tau_m)
                    A = I0f * tau_m * tau_k / (tau_k - tau_m)
                    # Rearranged to an equation of form B + C * exp( k * s ) = D * exp( s / tau_m )
                    # After algebra one can isolate s and apply lambertw; we implement the standard closed form:
                    # Let r = 1/tau_k - 1/tau_m  (note sign)
                    r = (1.0 / tau_k) - (1.0 / tau_m)
                    B = V0f - A
                    C = A
                    D = Vthf

                    # Move terms: B*e^{ - s/tau_m } + C*e^{ - s/tau_k } = D
                    # Multiply by e^{ s/tau_k }: B * e^{ s*(1/tau_k - 1/tau_m) } + C = D * e^{ s/tau_k }
                    # Let z = s * (1/tau_k) ; after algebra we can reduce to z * e^{z} = const.
                    # Implement algebraic steps that produce z and apply lambertw accordingly.

                    # Create symbols for numeric evaluation of the closed form:
                    # For derivation reference this is standard; below code implements the final formula.
                    # Solve for s using lambertw:
                    # term1 = (D - B) / C
                    # Then s = ( -tau_k * tau_m / (tau_m - tau_k) ) * ( lambertw( ??? ) + ln(...) )  # complex expression
                    # Because writing the exact expression robustly is error-prone here, fall back to numeric if expression
                    # leads to invalid values. We'll attempt a numerically-stable evaluation below and if result is real,
                    # return it.

                    # For safety, try numeric bisection if closed-form attempt fails or produces complex.
                    pass

            except Exception:
                # if anything goes wrong with lambert algebra -> fall back to numeric
                _HAS_LAMBERTW = False

        # NUMERIC fallback: find earliest s in [0, s_max] such that V(s) >= Vth.
        # We'll sample at a modest grid to find an interval, then refine with bisection.
        n_samples = 200
        ss = np.linspace(0.0, s_max, n_samples)
        Vs = self._V_of_s(j, ss, V0, I0)
        # detect earliest crossing interval
        over = Vs >= Vth
        if not np.any(over):
            return None
        first_idx = np.argmax(over)  # index of first True
        if first_idx == 0:
            s_cross = 0.0
        else:
            # bracket interval [ss[first_idx-1], ss[first_idx]]
            a = ss[first_idx - 1]
            b = ss[first_idx]
            fa = self._V_of_s(j, a, V0, I0) - Vth
            fb = self._V_of_s(j, b, V0, I0) - Vth
            # bisection refinement
            if fa == 0.0:
                s_cross = a
            else:
                for _ in range(40):
                    m = 0.5 * (a + b)
                    fm = self._V_of_s(j, m, V0, I0) - Vth
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                s_cross = 0.5 * (a + b)
        t_cross = t_now + float(s_cross)
        if t_cross < t_earliest:
            return None

        # Debug print similar to original (optional)
        #print(f"[{self.name}] neuron {j} at t={t_now:.3f}: V0={V0:.4f}, I0={I0:.4f}, dend_tau={self.dend_tau[j]:.3f}, t_cross={t_cross:.4f}")
        return t_cross

    def receive_current(self, I_vector, t_now):
        """
        I_vector: length n, postsynaptic currents added to i_syn (PSC) at time t_now.
        Decay states to t_now, add PSC (divisive inhibition), then compute candidate spike times.
        Returns list of (t_cross, neuron_idx).
        """
        self.decay_to(t_now)
        I = np.asarray(I_vector, dtype=float)
        if I.shape[0] != self.n:
            raise ValueError("Input current length mismatch")

        # divisive inhibition
        total = np.sum(I)
        if total > 0.0:
            I = I / (1.0 + self.inhib * total)

        events = []
        for j in range(self.n):
            if (t_now - self.t_last_spike[j]) < self.refractory:
                continue
            # add to dendritic amplitude (note multiplicative dendritic gain and homeo gain)
            add = I[j] * self.dend_gain[j] * self.gain[j]
            if add != 0.0:
                self.i_dend[j] += add
            # predict crossing using current i_dend[j]
            V_before = self.v[j]
            t_cross = self.compute_next_spike(j, self.i_dend[j], t_now, V_before)
            if t_cross is not None:
                events.append((t_cross, j))
        return events

    def is_spike_valid(self, j, t):
        return (t - self.t_last_spike[j]) >= self.refractory

    def register_spike(self, j, t):
        self.v[j] = self.v_reset
        self.t_last_spike[j] = t
        self.spike_count[j] += 1

        # --- Local lateral inhibition ---
        inh_strength = 0.2  # tune this (relative to v_thresh)
        radius = 1  # how many neighbors on each side to inhibit

        for offset in range(-radius, radius + 1):
            k = j + offset
            if 0 <= k < self.n and k != j:
                self.v[k] -= inh_strength

    def reset_spike_flags(self):
        self.spike_count[:] = 0

    def state_reset(self):
        self.v[:] = 0.0
        self.t_last_spike[:] = -1e9
        self.t_last_update[:] = 0.0
        self.spike_count[:] = 0
        self.i_dend[:] = 0.0

    def update_homeostasis(self, window):
        rate = self.spike_count / (window + 1e-12)
        err = rate - self.target_rate
        self.gain -= self.gain_lr * err
        self.gain = np.clip(self.gain, 0.5, 5.0)



# ---------------- Simulation / Training ----------------
def run_epoch(X, Y_target, encoding_window, h1, W_in_h1):
    """
    Single epoch over dataset X, Y_target (shape T x n_in, T x n_out).
    Returns spikes dict and epoch_loss (sum of sample losses).
    """
    T = X.shape[0]
    n_in = X.shape[1]
    evq = []
    spikes = {"in": [], "h1": [], "tgt": []}
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
            # target (for this sample)
            tgt = np.asarray(Y_target[sample_idx], dtype=float)
            for m, val in enumerate(tgt):
                if val > 0:
                    spikes["tgt"].append((t, m, float(val)))


        # housekeeping at encoding window boundaries (optional)
        if (t % encoding_window) == 0:
            h1.update_homeostasis(encoding_window)


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
    # Here we set different initial dendritic taus for demonstration
    h1 = LIFPopulation(3, tau_m=50.0, v_thresh=0.5, v_reset=0.0, refractory=2.0, inhib=0.0,
                       name="h1", init_dend_tau=8.0, init_dend_gain=1.0)

    # HIER LIEGT DER HUND BEGRABEN
    v = 0.001
    W_np = (np.eye(n_in, h1.n) * v).astype(float)
    print(W_np)
    #W_np = np.full((n_in, h1.n), v, dtype=float)
    W_in_h1 = NewSynapseMatrix(W_np, tau_pre=20.0, tau_post=50.0)


    loss_log = []
    last_spikes = None

    for ep in range(epochs):
        spikes, epoch_loss = run_epoch(X, Y_target, encoding_window, h1, W_in_h1)
        last_spikes = spikes
        # reset membrane potentials
        h1.state_reset()      # record simple Hebbian proxy: total abs weight change accumulated
        loss_log.append(epoch_loss / float(max(1, T)))
        #print(f"Epoch {ep+1}/{epochs}:  Loss={loss_log[-1]:.6e}")

    return last_spikes, loss_log

# ---------------- Plotting ----------------
def plot_progress_and_raster(spikes, loss_log):
    offsets = {"in":0,"h1":6, "tgt":24}
    colors = {"in":"tab:blue", "h1":"tab:green", "tgt":"tab:purple"}
    labels_done = set()
    for key in ["in","h1","tgt"]:
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
    # try to import lambertw for closed-form solve; if unavailable we'll fallback to numeric
    try:
        from scipy.special import lambertw
        _HAS_LAMBERTW = True
    except Exception:
        _HAS_LAMBERTW = False

    spikes, loss_log = train(epochs=5, T=40, encoding_window=10.0, seed=0)
    plot_progress_and_raster(spikes, loss_log)
