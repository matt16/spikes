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
def exp_decay_inplace(arr: np.ndarray, dt: float, tau: float):
    if dt <= 0.0 or tau <= EPS: return
    arr *= math.exp(-dt / tau)

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
    def __init__(self, n: int, tau_m: float=20.0, v_thresh: float=1.0,
                 v_reset: float=0.0, refractory: float=2.0,
                 inhib: float=0.2, thr_adapt: float=0.4, thr_tau: float=30.0,
                 name: str="pop"):
        self.name = name
        self.n = int(n)
        self.tau_m = float(tau_m)
        self.v_reset = float(v_reset)
        self.refractory = float(refractory)
        self.inhib = float(inhib)
        # Membrane and timing
        self.v = np.zeros(self.n, dtype=np.float64)
        self.t_last = 0.0
        self.t_last_spike = -1e9 * np.ones(self.n, dtype=np.float64)
        # Adaptive threshold
        self.vth_base = float(v_thresh)
        self.thr = np.full(self.n, self.vth_base, dtype=np.float64)
        self.thr_adapt = float(thr_adapt)
        self.thr_tau = float(thr_tau)

    def decay_to(self, t: float):
        dt = t - self.t_last
        if dt > 0.0:
            exp_decay_inplace(self.v, dt, self.tau_m)
            # thresholds relax back to baseline
            self.thr = self.vth_base + (self.thr - self.vth_base) * math.exp(-dt / max(self.thr_tau, EPS))
            self.t_last = t

    def receive_current(self, I: np.ndarray):
        self.v += I

    def apply_global_inhibition(self, n_spikes: int):
        if n_spikes > 0:
            self.v -= self.inhib * n_spikes

    def threshold_spikes(self, t: float) -> np.ndarray:
        """Return indices that spike at global time t; enforce refractory, reset & adapt."""
        spiking = []
        for j in range(self.n):
            # clamp while refractory
            if (t - self.t_last_spike[j]) < self.refractory:
                # keep near reset (prevents drift above thr)
                self.v[j] = self.v_reset
                continue
            if self.v[j] >= self.thr[j]:
                spiking.append(j)
                self.v[j] = self.v_reset
                self.thr[j] += self.thr_adapt      # spike-triggered threshold boost
                self.t_last_spike[j] = t
        if spiking:
            return np.array(spiking, dtype=np.int64)
        return np.empty(0, dtype=np.int64)

    def reset(self):
        self.v[:] = self.v_reset
        self.thr[:] = self.vth_base
        self.t_last = 0.0
        self.t_last_spike[:] = -1e9

# ---------------------------- Synapse with STDP + Oja ----------------------------
class SynapseMatrix:
    """pre -> post with trace STDP and Oja stabilization (all local)."""
    def __init__(self, n_pre: int, n_post: int,
                 A_plus: float=0.1, A_minus: float=0.12, alpha: float=1e-3,
                 tau_pre: float=10.0, tau_post: float=10.0,
                 w_init_scale: float=0.1, w_clip: Tuple[float,float]=(0.0, 1.0),
                 learn: bool=True, seed: Optional[int]=None, name: str="syn"):
        self.name = name
        self.n_pre, self.n_post = int(n_pre), int(n_post)
        self.learn = bool(learn)
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(0.0, w_init_scale, size=(self.n_pre, self.n_post)).astype(np.float64)
        self.pre_trace = ExpTrace(self.n_pre, tau_pre)
        self.post_trace = ExpTrace(self.n_post, tau_post)
        self.A_plus, self.A_minus, self.alpha = float(A_plus), float(A_minus), float(alpha)
        self.w_clip = w_clip


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
            np.clip(self.W[i, :], self.w_clip[0], self.w_clip[1], out=self.W[i, :])
        return self.W[i, :].copy()  # delivered current

    def on_post_spike(self, j: int, t: float):
        self.decay_to(t)
        self.post_trace.add(j, 1.0)
        if self.learn:
            x = self.pre_trace.val
            self.W[:, j] += -self.A_minus * x - self.alpha * (self.post_trace.val[j] ** 2) * self.W[:, j]
            # TODO: this is equal to self.W[:, j] += -self.A_minus * x - self.alpha * (1**2) * self.W[:, j]
            np.clip(self.W[:, j], self.w_clip[0], self.w_clip[1], out=self.W[:, j])

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
        self.h1 = LIFPopulation(n_h1, tau_m=params.get('tau_m', 20.0), v_thresh=params.get('v_thresh', 0.1), v_reset=params.get('v_reset', 0.0), refractory=params.get('refractory', 2.0), inhib=params.get('inhib', 0.2), thr_adapt=params.get('thr_adapt', 0.2), thr_tau=params.get('thr_tau', 50.0), name='h1')
        self.h2 = LIFPopulation(n_h2, tau_m=params.get('tau_m', 20.0), v_thresh=params.get('v_thresh', 0.1), v_reset=params.get('v_reset', 0.0), refractory=params.get('refractory', 2.0), inhib=params.get('inhib', 0.2), thr_adapt=params.get('thr_adapt', 0.2), thr_tau=params.get('thr_tau', 50.0), name='h2')
        self.out = LIFPopulation(n_out, tau_m=params.get('tau_m', 20.0), v_thresh=params.get('v_thresh', 0.1), v_reset=params.get('v_reset', 0.0), refractory=params.get('refractory', 2.0), inhib=params.get('inhib', 0.0), thr_adapt=params.get('thr_adapt', 0.2), thr_tau=params.get('thr_tau', 50.0), name='out')

        # Synapses
        #TODO: prevent synchronicity of neuron firing...
        self.in_h1  = SynapseMatrix(n_in,  n_h1,  **params.get('syn_in',  {}), name='in->h1')
        self.h1_rec = SynapseMatrix(n_h1,  n_h1,  **params.get('syn_rec1', {}), name='h1->h1')
        self.h1_h2  = SynapseMatrix(n_h1,  n_h2,  **params.get('syn_12',   {}), name='h1->h2')
        self.h2_rec = SynapseMatrix(n_h2,  n_h2,  **params.get('syn_rec2', {}), name='h2->h2')

        #TODO: include reward function
        self.syn_h2out = SynapseMatrix(n_h2, n_out, **params.get('syn_rec2', {}), name='h2->out')

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
    def run_stream(self, x_batch, y_batch, encoding_window=5.0, d_ax=0.5):
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
            print('h1', new_h1)
            if new_h1 is not None and len(new_h1) > 0:
                # optionally apply inhibition
                self.h1.apply_global_inhibition(len(new_h1))
                for j in new_h1:
                    spikes['h1'].append((self.t_global, int(j)))
                    heapq.heappush(evq, Event(self.t_global + d_ax, 'h1', int(j)))

            v_pre_h2 = self.h2.v.copy()
            new_h2 = self.h2.threshold_spikes(t)
            print('h2',new_h2,self.t_global)
            print('eventqueue', evq)
            if new_h2 is not None and len(new_h2) > 0:
                cand = np.asarray(new_h2, dtype=int)
                #k_wta: only the k neurons with the highest membrane potential are allowed to fire -> sparsity
                k2 = int(1)
                winners_h2 = cand if cand.size <= k2 else cand[np.argpartition(-v_pre_h2[cand], k2 - 1)[:k2]]
                # optionally apply inhibition
                self.h2.apply_global_inhibition(cand.size)
                for i in winners_h2:
                    spikes['h2'].append((self.t_global, int(i)))
                    heapq.heappush(evq, Event(self.t_global + d_ax, 'h2', int(i)))

            new_out = self.out.threshold_spikes(t)
            if new_out is not None and len(new_out) > 0:
                for j in new_out:
                    spikes["out"].append((self.t_global, int(j)))
                    if sample_idx not in pending_out:
                        pending_out[sample_idx] = t
                    heapq.heappush(evq, Event(self.t_global + d_ax, 'out', int(j)))

            ###PROBLEM: no spike -> 0 Loss: trains not to spike:(
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
    np.random.seed(50)
    n_epochs, B = 25, 20
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

