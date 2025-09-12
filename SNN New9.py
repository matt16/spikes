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
    Synapse matrix with:
      - pre_trace: exponential PSC per presynaptic neuron (drives postsyn currents)
      - post_trace: exponential trace for postsynaptic activity (learning)
      - pre_last/post_last: last spike times for STDP ordering decisions
      - update(pre_idx, post_idx, anti_hebb=...) applies Hebb/Oja or anti-Hebb
    """
    def __init__(self, n_pre, n_post, eta=1e-3, tau_pre=20.0, tau_post=50.0,
                 oja=True, is_output=False, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(0.5, 1.0, size=(n_pre, n_post))
        self.eta = float(eta)
        self.tau_pre = float(tau_pre)
        self.n_pre = n_pre
        self.n_post = n_post
        self.tau_post = float(tau_post)
        self.oja = bool(oja)
        self.is_output = bool(is_output)

        # traces & timing
        self.pre_trace = np.zeros(n_pre, dtype=float)   # PSC-like trace per presyn
        self.post_trace = np.zeros(n_post, dtype=float) # learning trace per post
        self.pre_last = -1e9 * np.ones(n_pre, dtype=float)
        self.post_last = -1e9 * np.ones(n_post, dtype=float)
        self.t_last = 0.0

        # logging
        self.last_dw_sum = 0.0

    def decay_to(self, t):
        dt = t - self.t_last
        if dt > 0:
            exp_decay_inplace(self.pre_trace, dt, self.tau_pre)
            exp_decay_inplace(self.post_trace, dt, self.tau_post)
            self.t_last = t

    def on_pre_spike(self, pre_idx, t, x_t=1.0):
        """Called when presynaptic neuron spikes (or receives analog input).
           Returns vector of postsynaptic currents (length n_post)."""
        self.decay_to(t)
        self.pre_trace[pre_idx] += x_t
        self.pre_last[pre_idx] = t
        # postsynaptic current for each post neuron = sum_over_pre W[pre,post] * pre_trace[pre]
        # efficient: vector multiply pre_trace @ W -> length n_post
        return self.pre_trace @ self.W

    def on_post_spike(self, post_idx, t):
        """Called when a postsynaptic neuron spikes (for learning trace)."""
        self.decay_to(t)
        self.post_trace[post_idx] += 1.0
        self.post_last[post_idx] = t

    def update(self, pre_idx, post_idx, reward=None, anti_hebb=False):
        """Update single synapse W[pre_idx, post_idx]. Returns |dw| for logging."""
        lr = float(self.eta)
        if self.is_output and (reward is not None):
            # lr *= float(reward)
            lr *= 1
        sign = -1.0 if anti_hebb else 1.0
        dw = sign * lr * self.pre_trace[pre_idx] * self.post_trace[post_idx]
        if self.oja:
            dw -= lr * (self.post_trace[post_idx] ** 2) * self.W[pre_idx, post_idx]
        self.W[pre_idx, post_idx] += dw
        self.last_dw_sum += abs(dw)
        return abs(dw)

    def reset_dw_log(self):
        val = self.last_dw_sum
        self.last_dw_sum = 0.0
        return val

    def reset_traces(self):
        self.pre_trace = np.zeros(self.n_pre, dtype=float)   # PSC-like trace per presyn
        self.pre_post = np.zeros(self.n_post, dtype=float)  # PSC-like trace per presyn

# ---------------- LIF Population (PSC + membrane) ----------------
class LIFPopulation:
    """
    LIF with synaptic PSC trace (i_syn) and membrane V.
    i_syn decays with tau_syn; V decays with tau_m.
    compute_next_spike predicts crossing time assuming i_syn stays constant.
    """
    def __init__(self, n, tau_m=20.0, tau_syn=5.0, v_thresh=1.0, v_reset=0.0,
                 refractory=0.0, inhib=0.1, name="pop"):
        self.name = name
        self.n = n
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.refractory = refractory
        self.inhib = inhib

        # states
        self.v = np.zeros(n)               # membrane potentials
        self.t_last_spike = -1e9 * np.ones(n)  # last spike times
        self.gain = np.ones(n)
        self.gain_lr = 0 * 1e-3

        # homeostasis
        self.spike_count = np.zeros(n)
        self.target_rate = 0.05

        # track last update time per neuron
        self.t_last_update = np.zeros(n)

    def decay_to(self, t):
        dt = t - self.t_last_update
        mask = dt > 0
        if np.any(mask):
            self.v[mask] *= np.exp(-dt[mask] / self.tau_m)
            self.t_last_update[mask] = t

    def compute_next_spike(self, j, I_drive, t_now, V_before):
        """
        Predict next spike time for neuron j assuming I_drive (PSC) remains constant.
        Solve V(t) = V_inf + (V0 - V_inf) * exp(-(t - t_now)/tau_m) = V_thresh
        where V_inf = i_syn * tau_m.
        Return t_cross (absolute) or None.
        """
        t_earliest = max(t_now, self.t_last_spike[j] + self.refractory)

        V0 = V_before
        Vth = self.v_thresh
        V_inf = I_drive * self.tau_m  # steady-state voltage

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
        # dt cross
        dt_cross = - self.tau_m * safe_log(ratio)

        if not np.isfinite(dt_cross) or dt_cross < 0.0:
            return None
        t_cross = t_now + dt_cross
        if t_cross < t_earliest:
            return None

        print(f"[{self.name}] neuron {j} at t={t_now:.1f}: "
              f"I0={V0:.3f}, V_inf={V_inf:.3f}, Vth={self.v_thresh}, V0={self.v[j]:.3f}, t_cross={t_cross:.3f}")

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
            V_before = self.v[j]  # store before input
            self.v[j] += I[j] * self.gain[j]  # integrate
            t_cross = self.compute_next_spike(j, I[j], t_now, V_before)
            if t_cross is not None:
                events.append((t_cross, j))
        return events

    def is_spike_valid(self, j, t):
        return (t - self.t_last_spike[j]) >= self.refractory

    def register_spike(self, j, t):
        self.v[j] = self.v_reset
        self.t_last_spike[j] = t
        self.spike_count[j] += 1

    def reset_spike_flags(self):
        self.spike_count[:] = 0

    def state_reset(self):
        self.v = np.zeros(self.n)

    def update_homeostasis(self, window):
        rate = self.spike_count / (window + 1e-12)
        err = rate - self.target_rate
        self.gain -= self.gain_lr * err
        self.gain = np.clip(self.gain, 0.5, 5.0)



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

        # housekeeping at encoding window boundaries (optional)
        if (t % encoding_window) == 0:
            h1.update_homeostasis(encoding_window)
            h2.update_homeostasis(encoding_window)

    print(I_to_h1_reg)

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
    h1 = LIFPopulation(5, tau_m=50.0, tau_syn=5.0, v_thresh=0.5, v_reset=0.0, refractory=2.0, inhib=0.0, name="h1")
    h2 = LIFPopulation(4, tau_m=50.0, tau_syn=5.0, v_thresh=0.5, v_reset=0.0, refractory=2.0, inhib=0.0, name="h2")

    W_in_h1 = SynapseMatrix(n_in, h1.n, eta=0e-3, tau_pre=5.0, tau_post=30.0, oja=False, seed=1)
    W_h1_h2 = SynapseMatrix(h1.n, h2.n, eta=0e-3, tau_pre=5.0, tau_post=30.0, oja=False, seed=2)
    W_h2_out = SynapseMatrix(h2.n, n_out, eta=0e-3, tau_pre=5.0, tau_post=50.0, oja=False, is_output=True, seed=3)

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
    spikes, hebb_log, loss_log = train(epochs=2, T=40, encoding_window=10.0, seed=0)
    plot_progress_and_raster(spikes, hebb_log, loss_log)
