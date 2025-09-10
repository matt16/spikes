import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# ---------------- Event ----------------
Event = namedtuple("Event", ["time", "src", "idx"])

# ---------------- Helpers ----------------
EPS = 1e-12
def exp_decay_inplace(arr, dt, tau):
    if tau > EPS:
        arr *= np.exp(-dt / tau)

# ---------------- Synapse matrix ----------------
class SynapseMatrix:
    def __init__(self, n_pre, n_post, eta=1e-3, oja=True, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(0.5, 1.0, size=(n_pre, n_post))
        self.eta = eta
        self.oja = oja

    def transmit(self, pre_idx, x_t=1.0):
        """
        Compute postsynaptic currents.
        pre_idx: presynaptic neuron index
        x_t: input magnitude (analog sample value)
        """
        return x_t * self.W[pre_idx, :]

    def update(self, pre_idx, post_spikes):
        """
        Hebb + Oja learning rule (not applied in this demo).
        pre_idx: presynaptic neuron index (int)
        post_spikes: binary vector [n_post]
        """
        if self.eta <= 0:
            return
        pre_vec = np.zeros(self.W.shape[0])
        pre_vec[pre_idx] = 1.0
        for j in range(self.W.shape[1]):
            dw = self.eta * pre_vec[pre_idx] * post_spikes[j]
            if self.oja:
                dw -= self.eta * (post_spikes[j] ** 2) * self.W[pre_idx, j]
            self.W[pre_idx, j] += dw

# ---------------- LIF population ----------------
class LIFPopulation:
    def __init__(self, n, tau_m=20.0, v_thresh=1.0, v_reset=0.0,
                 refractory=2.0, inhib=0.1, name="pop"):
        self.name = name
        self.n = n
        self.tau_m = tau_m
        self.vth_base = v_thresh
        self.v_reset = v_reset
        self.refractory = refractory
        self.inhib = inhib

        # states
        self.v = np.zeros(n)
        self.thr = np.full(n, self.vth_base)
        self.t_last = np.zeros(n)
        self.t_last_spike = -1e9 * np.ones(n)
        self.next_spike_time = np.full(n, np.inf)

    def decay_neurons_to(self, idx, t):
        idx = np.atleast_1d(idx)
        dt = t - self.t_last[idx]
        mask = dt > 0
        if np.any(mask):
            self.v[idx[mask]] *= np.exp(-dt[mask] / self.tau_m)
            self.t_last[idx[mask]] = t

    def compute_next_spike(self, V0, thr, t_now, t_last_spike):
        if (t_now - t_last_spike) < self.refractory:
            return None
        V_inf = 0.0
        if V0 <= thr:
            return None
        dt_cross = -self.tau_m * np.log((thr - V_inf) / (V0 - V_inf))
        if dt_cross <= 0:
            return None
        return t_now + dt_cross

    def receive_current(self, I, t):
        idx = np.arange(self.n)
        self.decay_neurons_to(idx, t)
        # divisive inhibition
        total = np.sum(I)
        if total > 0:
            I = I / (1.0 + self.inhib * total)
        self.v[idx] += I

        events = []
        for j in idx:
            t_cross = self.compute_next_spike(self.v[j], self.thr[j], t, self.t_last_spike[j])
            if t_cross is not None:
                self.next_spike_time[j] = t_cross
                events.append((t_cross, j))
            else:
                self.next_spike_time[j] = np.inf
        return events

    def is_spike_valid(self, j, t_event):
        return np.isclose(self.next_spike_time[j], t_event, rtol=1e-9, atol=1e-9)

    def register_spike(self, idxs, t):
        idxs = np.atleast_1d(idxs)
        self.decay_neurons_to(idxs, t)
        self.v[idxs] = self.v_reset
        self.thr[idxs] = self.vth_base
        self.t_last_spike[idxs] = t
        self.next_spike_time[idxs] = np.inf

# ---------------- Network setup ----------------
n_in, n_h1, n_h2 = 3, 5, 4
h1 = LIFPopulation(n_h1, name="h1")
h2 = LIFPopulation(n_h2, name="h2")

W_in_h1 = SynapseMatrix(n_in, n_h1, eta=1e-3, oja=True)
W_h1_h2 = SynapseMatrix(n_h1, n_h2, eta=1e-3, oja=True)

# ---------------- Event queue ----------------
evq = []
spikes = {"in": [], "h1": [], "h2": []}

# input spikes at different times
input_times = [1.0, 1.5, 2.0]
for idx, t in enumerate(input_times):
    heapq.heappush(evq, Event(time=t, src="in", idx=idx))

# ---------------- Simulation loop ----------------
while evq:
    ev = heapq.heappop(evq)
    t = ev.time

    if ev.src == "in":
        spikes["in"].append((t, ev.idx))
        x_t = np.random.uniform(0.8, 1.2)  # analog input magnitude
        I = W_in_h1.transmit(ev.idx, x_t)
        new_events = h1.receive_current(I, t)
        for t_cross, j in new_events:
            heapq.heappush(evq, Event(time=t_cross, src="h1", idx=j))

    elif ev.src == "h1":
        j = ev.idx
        if not h1.is_spike_valid(j, t):
            continue
        h1.register_spike(j, t)
        spikes["h1"].append((t, j))
        I = W_h1_h2.transmit(j, 1.0)
        new_events = h2.receive_current(I, t)
        for t_cross, k in new_events:
            heapq.heappush(evq, Event(time=t_cross, src="h2", idx=k))

    elif ev.src == "h2":
        j = ev.idx
        if not h2.is_spike_valid(j, t):
            continue
        h2.register_spike(j, t)
        spikes["h2"].append((t, j))

# ---------------- Raster plot ----------------
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 4))

# H1
times = [t for t, nid in spikes["h1"]]
ids   = [nid for t, nid in spikes["h1"]]
ax[0].scatter(times, ids, c="b", s=40)
ax[0].set_ylabel("H1")
ax[0].set_yticks(range(h1.n))

# H2
times = [t for t, nid in spikes["h2"]]
ids   = [nid for t, nid in spikes["h2"]]
ax[1].scatter(times, ids, c="r", s=40)
ax[1].set_ylabel("H2")
ax[1].set_yticks(range(h2.n))
ax[1].set_xlabel("Time")

plt.suptitle("Raster plot: H1 (blue) and H2 (red)")
plt.show()