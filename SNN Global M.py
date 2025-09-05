# ===== Event-driven Deep Causal SNN (global time, analog-friendly) =====
#   - Latency encoding with fixed window
#   - Event-driven LIF with refractory + adaptive thresholds
#   - STDP (pre/post traces) + Oja stabilization
#   - Global time stream: run_stream(X, Y)
#   - numpy + heapq only
from __future__ import annotations
import math, heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

EPS = 1e-12

# ---------------------------- Utilities ----------------------------
def exp_decay_inplace(arr, dt, tau):
    if dt <= 0.0: return
    arr *= np.exp(-dt / np.maximum(tau, EPS))  # vector of decays

class ExpTrace:
    """Exponentially-decaying trace; event-driven."""
    def __init__(self, size: int, tau: float):
        self.val = np.zeros(size, dtype=np.float64)
        self.tau = float(tau)
        self.t_last = 0.0

    def decay_to(self, t: float):
        dt = t - self.t_last
        if dt > 0.0:
            exp_decay_inplace(self.val, dt, self.tau)
            self.t_last = t

    def add(self, idx: Optional[int] = None, amount: float = 1.0):
        if idx is None:
            self.val += amount
        else:
            self.val[idx] += amount

# ---------------------------- Events ----------------------------
@dataclass(order=True)
class Event:
    time: float
    src: str   # 'in', 'h1', 'h2', 'target'
    idx: int

# ---------------------------- LIF Population ----------------------------
class LIFPopulation:
    """Event-driven LIF with adaptive threshold + refractory; global time aware."""
    def __init__(self, n, tau_m=20.0, v_thresh=1.0, v_reset=0.0, refractory=2.0, inhib=0.1, thr_adapt=0.2, thr_tau=50.0, name="pop"):
        self.name = name
        self.n = int(n)
        self.tau_m = float(tau_m)
        # rng = np.random.default_rng(0)
        # self.tau_m = rng.uniform(5.0, 50.0, size=self.n)  # ms
        self.v_reset = float(v_reset)
        self.refractory = float(refractory)
        self.inhib = float(inhib)
        # Membrane and timing
        self.v = np.zeros(self.n, dtype=np.float64)
        self.t_last = 0.0
        self.t_last_spike = -1e9 * np.ones(self.n, dtype=np.float64)
        self.spike_count = np.zeros(n, dtype=np.float64)  # for rate estimation

        # Adaptive threshold
        self.vth_base = float(v_thresh)
        self.thr = np.full(self.n, self.vth_base, dtype=np.float64)
        self.thr = self.vth_base + np.random.normal(0, 0.5, size=self.n)
        # self.thr_adapt = float(thr_adapt)
        self.thr_tau = float(thr_tau)

        # Intrinsic plasticity (IP)
        self.rate_trace = ExpTrace(self.n, tau=200.0)  # very slow trace (longer than encoding_window)
        self.gain = np.ones(self.n)  # per-neuron multiplicative gain
        self.target_rate = 0.05  # spikes per encoding window (tune this!)
        self.gain_lr = 1e-4  # learning rate for gain updates

        # AHP variables
        self.ahp = np.zeros(self.n, dtype=np.float64)
        self.ahp_tau = 50.0       # decay time constant of AHP
        self.ahp_step = 0.1

    def decay_to(self, t: float):
        dt = t - self.t_last
        if dt > 0.0:
            exp_decay_inplace(self.v, dt, self.tau_m)
            # TODO: try with constant thr NOT: thresholds relax back to baseline
            self.thr = self.vth_base + (self.thr - self.vth_base) * math.exp(-dt / max(self.thr_tau, EPS))
            self.t_last = t
            # AHP decay
            exp_decay_inplace(self.ahp, dt, self.ahp_tau)
            self.t_last = t

    def receive_current(self, I):
        # Apply divisive inhibition: I / (1 + k * sum(I))
        total = np.sum(I)
        if total > 0:
            I = I / (1.0 + self.inhib * total)
        self.v += I * self.gain
        # self.v += I

    def threshold_spikes(self, t: float) -> np.ndarray:
        """Return indices that spike at global time t; enforce refractory, reset & adapt."""
        spiking = []
        for j in range(self.n):
            # clamp while refractory
            if (t - self.t_last_spike[j]) < self.refractory:
                # keep near reset (prevents drift above thr)
                self.v[j] = self.v_reset
                continue
            # effective_v = self.v[j] - self.ahp[j]
            effective_v = self.v[j]
            if effective_v >= self.thr[j]:
                spiking.append(j)
                self.v[j] = self.v_reset
                # self.thr[j] += self.thr_adapt      # spike-triggered threshold boost
                self.ahp[j] += self.ahp_step        # AHP increment
                self.t_last_spike[j] = t
                self.spike_count[j] += 1
        # --- intrinsic plasticity update ---
        # err = self.spike_count / max(t - self.t_last + 1e-12, 1e-12) - self.target_rate
        # self.gain -= self.gain_lr * err
        # self.gain = np.clip(self.gain, 1.0, 5.0)
        # # normalize gains across population
        # self.gain /= np.mean(self.gain)
        self.gain = 1.0
        if spiking:
            return np.array(spiking, dtype=np.int64)
        return np.empty(0, dtype=np.int64)

    def reset(self):
        self.v[:] = self.v_reset
        self.thr[:] = self.vth_base + np.random.normal(0, 0.5, size=self.n)
        # self.thr[:] = self.vth_base
        self.t_last = 0.0
        self.t_last_spike[:] = -1e9

# ---------------------------- Synapse with STDP + Oja ----------------------------
class SynapseMatrix:
    def __init__(self, n_pre, n_post, A_plus=0.01, A_minus=0.012, alpha=0.001, tau_pre=20.0, tau_post=20.0, dale_ratio=0.8, g_exc=1.0, g_inh=0.3, learn=True, name ="syn"):
        self.name = name
        self.n_pre = n_pre
        self.n_post = n_post
        self.learn = learn
        # ---- Dale’s law ----
        # Randomly assign excitatory (+1) or inhibitory (-1) for each presynaptic neuron
        signs = -1.0 * np.ones(n_pre)
        n_exc = max(1, int(dale_ratio * self.n_pre))
        exc_idx = np.random.choice(n_pre, size=n_exc, replace=False)
        signs[exc_idx] = 1.0  # excitatory
        self.dale_signs = signs  # save for later use
        # Random initial weights
        W = np.random.rand(n_pre, n_post) * 0.1
        # add tiny floor to excitatory rows so they’re not all ~0
        W[exc_idx, :] += 0.02
        # apply Dale signs and row gains
        row_gain = np.where(self.dale_signs > 0.0, g_exc, g_inh)[:, None]
        # self.W = (W * self.dale_signs[:, None]) * row_gain
        self.W = (W * self.dale_signs[:, None])
        # Hebbian/Oja learning params
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.alpha = alpha
        # Traces
        self.pre_trace = ExpTrace(n_pre, tau=tau_pre)
        self.post_trace = ExpTrace(n_post, tau=tau_post)

    def decay_to(self, t: float):
        self.pre_trace.decay_to(t)
        self.post_trace.decay_to(t)

    def on_pre_spike(self, i: int, t: float) -> np.ndarray:
        self.decay_to(t)
        self.pre_trace.add(i, 1.0)
        if self.learn:
            y = self.post_trace.val  # latency-aware post activity
            # Hebb (+) and Oja decay
            # TODO: W = (1-y^2) W + A_plus * y -> is like like a moving avg over A_plus * y (VERIFY)
            self.W[i, :] += self.A_plus * y - self.alpha * (y**2) * self.W[i, :]
            self._dale_clip()
        return self.W[i, :].copy()  # delivered current

    def on_post_spike(self, j: int, t: float):
        self.decay_to(t)
        self.post_trace.add(j, 1.0)
        if self.learn:
            x = self.pre_trace.val
            # TODO: this is equal to self.W[:, j] += -self.A_minus * x - self.alpha * (1**2) * self.W[:, j]
            self.W[:, j] += -self.A_minus * x - self.alpha * (self.post_trace.val[j] ** 2) * self.W[:, j]
            self._dale_clip()

    def _dale_clip(self):
        """Enforce Dale’s law by clipping weights to correct sign."""
        for i in range(self.n_pre):
            exc_rows = self.dale_signs > 0.0
            inh_rows = ~exc_rows
            if np.any(exc_rows):
                self.W[exc_rows, :] = np.clip(self.W[exc_rows, :], 0.0, None)
            if np.any(inh_rows):
                self.W[inh_rows, :] = np.clip(self.W[inh_rows, :], None, 0.0)

    def reset(self):
        self.pre_trace.val[:] = 0.0
        self.pre_trace.t_last = 0.0
        self.post_trace.val[:] = 0.0
        self.post_trace.t_last = 0.0


# ---------------------------- Latency (time-to-first-spike) Encoder ----------------------------
class LatencyEncoder:
    """Map values in [0,1] to spike times in [t_offset + t_eps, t_offset + window].
       Higher value -> earlier spike. Value<=0 -> no spike.
    """
    def __init__(self, window: float=20.0, t_eps: float=1e-3):
        self.window = float(window)
        self.t_eps = float(t_eps)

    def encode(self, x: np.ndarray, t_offset: float=0.0) -> List[Tuple[float, int]]:
        ev = []
        w, te = self.window, self.t_eps
        for i, v in enumerate(np.asarray(x, dtype=float)):
            if v <= 0.0: continue
            t = t_offset + te + (1.0 - v) * (w - te)
            ev.append((t, i))
        return ev

# ---------------------------- Network (global time) ----------------------------
class DeepCausalSNN_Event:
    def __init__(self, n_in: int, n_h1: int, n_h2: int, n_out: int, params: dict):

        # Populations
        self.h1 = LIFPopulation(n_h1, tau_m = 40.0, v_thresh= 0.1, v_reset= 0.0, refractory=1.0, inhib=0.1, thr_adapt=0.1, thr_tau=30.0, name='h1')
        self.h2 = LIFPopulation(n_h2, tau_m = 40.0, v_thresh= 0.1, v_reset= 0.0, refractory=1.0, inhib=0.1, thr_adapt=0.1, thr_tau=30.0, name='h2')
        self.out = LIFPopulation(n_out, tau_m= 40.0, v_thresh= 0.1, v_reset=0.0, refractory=1.0, inhib=0.1, thr_adapt=0.1, thr_tau=30.0, name='out')

        # Synapses
        #TODO: prevent synchronicity of neuron firing...
        self.in_h1  = SynapseMatrix(n_in,  n_h1,  dale_ratio=1.0, g_exc=1.0, g_inh=0.2,  A_plus=0.03, A_minus=0.01, alpha=1e-4, name='in->h1')
        self.h1_rec = SynapseMatrix(n_h1,  n_h1,  dale_ratio=0.8,  g_exc=0.7, g_inh=0.7,  A_plus=0.01, A_minus=0.015,alpha=2e-4, name='h1->h1')
        self.h1_h2  = SynapseMatrix(n_h1,  n_h2,  dale_ratio=0.8,  g_exc=1.0, g_inh=0.3,  A_plus=0.02, A_minus=0.01, alpha=1e-4, name='h1->h2')
        self.h2_rec = SynapseMatrix(n_h2,  n_h2,  dale_ratio=0.8,  g_exc=0.7, g_inh=0.7,  A_plus=0.01, A_minus=0.015,alpha=2e-4, name='h2->h2')

        #TODO: include reward function
        self.syn_h2out = SynapseMatrix(n_h2,  n_out, dale_ratio=1.0,  g_exc=1.0, g_inh=0.0,  A_plus=0.02, A_minus=0.0,  alpha=1e-4, name='h2->out')

        # Encoders
        self.enc_in  = LatencyEncoder(window=params['encoding_window'])
        self.enc_out = LatencyEncoder(window=params['encoding_window'])

        # Global clock
        self.t_global = 0.0
        self.vth_out = 0.1
        self.sample_window = float(params.get('encoding_window', 20.0))
        self.silent_gap = float(params.get('silent_gap', 0.0))  # idle time between samples

        #avg losses
        self.avg_latency_loss_per_epoch = []
        self._epoch_losses = []

    def epoch_reset(self):
        """Reset all time-dependent variables at epoch start; keep weights."""
        self.t_global = 0.0
        # Reset LIF neurons
        [pop.reset() for pop in [self.h1, self.h2, self.out]]
        # Reset synapse traces
        [syn.reset() for syn in [self.in_h1, self.h1_rec, self.h1_h2, self.h2_rec, self.syn_h2out]]

    # ---- core: process a stream of samples sequentially on the global clock ----
    def run_stream(self, x_batch, y_batch, encoding_window=50.0, d_ax=0.5):
        # Continuous event-driven run over a batch, using absolute global time.
        B = len(x_batch)
        evq = []
        spikes = {"in": [], "h1": [], "h2": [], "out": [], "target": []}
        # Encode all input + target spikes with absolute global time
        for b, (x, y) in enumerate(zip(x_batch, y_batch)):
            t_offset = b * encoding_window
            in_spikes = self.enc_in.encode(x, t_offset)
            target_spikes = self.enc_out.encode(y, t_offset)
            for (t, i) in in_spikes:
                heapq.heappush(evq, Event(t, 'in', i))
            for (t, j) in target_spikes:
                heapq.heappush(evq, Event(t, 'target', j))

        # Variables to collect per-sample latencies
        pending_out = {}
        pending_tgt = {}
        loss = []

        while evq:
            ev = heapq.heappop(evq)
            # Update global time
            self.t_global = ev.time
            t = ev.time
            # Decode sample index from global time
            sample_idx = int(t // encoding_window)
            # Decay to current global time
            [e.decay_to(t) for e in [self.h1, self.h2, self.in_h1, self.h1_rec, self.h1_h2, self.h2_rec, self.syn_h2out]]

            # Handle event types
            if ev.src == 'in':
                spikes["in"].append((t, ev.idx))
                I = self.in_h1.on_pre_spike(ev.idx, t)
                #TODO: on_pre_spike(ev.idx, t, spikes['in']_trailing_count)
                #TODO: trailing_count adjusts A
                self.h1.receive_current(I)

            elif ev.src == 'h1':
                spikes["h1"].append((t, ev.idx))
                j = ev.idx
                self.h1_rec.on_post_spike(j, t)
                Irec = self.h1_rec.on_pre_spike(j, t)
                self.h1.receive_current(Irec)
                I12 = self.h1_h2.on_pre_spike(j, t)
                self.h2.receive_current(I12)

            elif ev.src == 'h2':
                spikes["h2"].append((t, ev.idx))
                k = ev.idx
                self.h2_rec.on_post_spike(k, t)
                Irec2 = self.h2_rec.on_pre_spike(k, t)
                self.h2.receive_current(Irec2)
                Iout = self.syn_h2out.on_pre_spike(ev.idx, t)
                self.out.receive_current(Iout)

            elif ev.src == 'target':
                spikes["target"].append((t, ev.idx))
                self.syn_h2out.on_post_spike(ev.idx, t)
                if sample_idx not in pending_tgt:
                    pending_tgt[sample_idx] = t

            # After delivering currents, check for new spikes
            new_h1 = self.h1.threshold_spikes(t)
            if new_h1 is not None and len(new_h1) > 0:
                # optionally apply inhibition
                for j in new_h1:
                    spikes['h1'].append((self.t_global, int(j)))
                    # d_ax = np.random.uniform(0.5, 2.0)  # ms
                    heapq.heappush(evq, Event(self.t_global + d_ax, 'h1', int(j)))

            new_h2 = self.h2.threshold_spikes(t)
            if new_h2 is not None and len(new_h2) > 0:
                for j in new_h2:
                    spikes['h2'].append((self.t_global, int(j)))
                    # d_ax = np.random.uniform(0.5, 2.0)  # ms
                    heapq.heappush(evq, Event(self.t_global + d_ax, 'h2', int(j)))

            new_out = self.out.threshold_spikes(t)
            if new_out is not None and len(new_out) > 0:
                for j in new_out:
                    spikes["out"].append((self.t_global, int(j)))
                    if sample_idx not in pending_out:
                        pending_out[sample_idx] = t
                    # d_ax = np.random.uniform(0.5, 2.0)  # ms
                    heapq.heappush(evq, Event(self.t_global + d_ax, 'out', int(j)))

            # --- Loss computation (latency-based) ---
            if sample_idx in pending_out and sample_idx in pending_tgt:
                t_out = pending_out[sample_idx]
                t_tgt = pending_tgt[sample_idx]
                latency_loss = (t_out - t_tgt) ** 2
                # model._log_sample_latency_loss(latency_loss)  # <<< ADD (pro Sample)
                loss.append((self.t_global, (t_out - t_tgt) ** 2))
                del pending_out[sample_idx]
                del pending_tgt[sample_idx]

        return spikes, loss

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    np.random.seed(7)
    n_epochs, B = 5, 10
    N_IN, N_OUT = 2, 1
    t_max_per_sample = 50
    params = {"n_h1": 80, "n_h2": 60, "encoding_window": t_max_per_sample}

    # Generate deterministic mapping
    x = np.random.rand(B, N_IN)
    r = (x[:,0]-.5) ** 2 + (x[:,1]-.5) ** 2
    y = np.expand_dims((1 - r) * np.exp(-r /2),1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x[:, 0], x[:, 1], y[:, 0], c=y[:, 0], cmap='viridis', s=50)
    # plt.show(block=True)

    all_losses = []
    model = DeepCausalSNN_Event(N_IN, params["n_h1"], params["n_h2"], N_OUT, params)
    for epoch in range(n_epochs):
        print(epoch)
        model.epoch_reset()
        spikes, loss = model.run_stream(x, y, t_max_per_sample, d_ax=0.05)
        times, losses = zip(*loss) if loss else ([], [])
        all_losses.append((epoch, times, losses))

    plt.figure(figsize=(8, 5))
    epoch_loss = np.array([np.mean(l) for _, _, l in all_losses])
    plt.plot(range(n_epochs),epoch_loss , marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Durchschnittlicher Latency Loss")
    plt.title("Verlauf des durchschnittlichen Latency Loss pro Epoche")
    plt.grid(True)



    # ---------------------------- Raster Plot with Sample Boundaries ----------------------------
    def plot_spikes_raster(spikes, t_max_per_sample, n_samples, layer_order=None):
        if layer_order is None:
            layer_order = ["in", "h1", "h2", "out", "target"]

        fig, axes = plt.subplots(len(layer_order)+1, 1, sharex=True, figsize=(10, 8))
        for ax, key in zip(axes, layer_order):
            data = spikes[key]
            if data:
                t, n = zip(*data)
                ax.scatter(t, n, s=10, marker="|")
            ax.set_ylabel(key)
            ax.grid(True, linestyle="--", alpha=0.3)

            # Draw sample boundaries
            for s in range(1, n_samples):
                ax.axvline(s * t_max_per_sample, color="red", linestyle="--", alpha=0.5)

        axes[-1].set_xlabel("Global Time")

        if all_losses is not None:
            ax = axes[-1]
            times, losses = all_losses[0][1], all_losses[0][2]
            ax.plot(times, losses, label='first epoch')
            times, losses = all_losses[-1][1], all_losses[-1][2]
            ax.plot(times, losses, label='last epoch')
            # for epoch, times, losses in all_losses:
            #     if times and losses:
            #         ax.plot(times, losses, label=f"Epoch {epoch}")
            ax.set_ylabel("Latency loss")
            ax.set_xlabel("Global time")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    plot_spikes_raster(spikes, t_max_per_sample, B)

