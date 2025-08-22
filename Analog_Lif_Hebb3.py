"""
Self-contained event-driven DeepCausalSNN with:
- Latency encoding to spike times
- LIF neurons (event-driven), global inhibition
- Synapse matrices with trace-based STDP and Oja stabilization
- Two hidden layers + decoder trained with teacher spikes (Oja)

No fixed time-step loops: processing happens on spike events only.
Tested with Python 3.10+. Uses only numpy and heapq (no TF/PyTorch).
"""
from __future__ import annotations
import math
import heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt

# ---------------------------- Utilities ----------------------------
EPS = 1e-12

class ExpTrace:
    """Exponentially decaying trace x(t) with piecewise-constant updates."""
    def __init__(self, size: int, tau: float):
        self.val = np.zeros(size, dtype=np.float64)
        self.tau = float(tau)
        self.t_last = 0.0

    def decay_to(self, t: float):
        dt = max(0.0, t - self.t_last)
        if dt > 0.0:
            self.val *= math.exp(-dt / max(self.tau, EPS))
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
    src: str        # 'in', 'h1', 'h2', 'target'
    idx: int        # neuron index within src population

# ---------------------------- LIF Population ----------------------------

class LIFPopulation:
    #thresholds stabilize, weights encode
    def __init__(self, n, tau=20.0, v_th=0.5, v_reset=0.0, inhib = 0.2, refractory=1.0):
        self.n = n  # number of neurons in population
        self.tau = float(tau)  # membrane time constant
        self.v_th = float(v_th)  # threshold
        self.inhib = float(inhib)
        self.v_reset = float(v_reset)  # reset value
        self.refractory = refractory  # refractory period (same units as time)
        self.v = np.zeros(n)  # membrane potentials
        self.last_update = 0.0  # last time decay was applied
        self.last_spike = -1e9 * np.ones(n)  # last spike time for each neuron

    def decay_to(self, t):
        """Exponential decay of all membrane potentials to time t"""
        dt = t - self.last_update
        if dt > 0:
            self.v *= np.exp(-dt / self.tau)
            self.last_update = t

    def receive_current(self, I):
        """Add input current to membrane potential"""
        self.v += I

    def threshold_spikes(self, t):
        """Check which neurons spike at time t"""
        spikes = []
        for j in range(self.n):
            # If neuron is in refractory → clamp to reset
            if t - self.last_spike[j] < self.refractory:
                self.v[j] = self.v_reset
                continue
            # Normal spike condition
            if self.v[j] >= self.v_th:
                spikes.append((t, j))
                self.v[j] = self.v_reset
                self.last_spike[j] = t
        return spikes

    def apply_global_inhibition(self, n_spikes: int):
        if n_spikes > 0:
            self.v -= self.inhib * n_spikes

# ---------------------------- Synapse with STDP + Oja ----------------------------
class SynapseMatrix:
    """Synapses from pre -> post with trace STDP and Oja stabilization.

    Local state:
      - pre_trace (size N_pre), post_trace (size N_post)
      - weights W [N_pre, N_post]

    Update rules (event-driven):
      On pre spike i at time t:
        pre_trace[i] += 1
        DW += A_plus * pre[i]*post_trace[:] - alpha * (post_activity[:]**2) * W[i,:]
        deliver current to post: I_post += W[i,:]

      On post spike j at time t:
        post_trace[j] += 1
        DW += -A_minus * pre_trace[:] * post[j] - alpha * (post[j]**2) * W[:,j]

    where post_activity is taken as post_trace to make latency-aware (earlier spikes persist longer).
    """
    def __init__(self,
                 n_pre: int,
                 n_post: int,
                 A_plus: float = 0.01,
                 A_minus: float = 0.012,
                 alpha: float = 0.000,   # Oja coefficient
                 tau_pre: float = 20.0,
                 tau_post: float = 20.0,
                 w_init_scale: float = 0.1,
                 w_clip: Tuple[float, float] = (0.0, 1.0),
                 seed: Optional[int] = None,
                 learn: bool = True,
                 name: str = "syn"):
        self.name = name
        self.n_pre = n_pre
        self.n_post = n_post
        self.learn = learn
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(0.0, w_init_scale, size=(n_pre, n_post)).astype(np.float64)
        self.pre_trace = ExpTrace(n_pre, tau_pre)
        self.post_trace = ExpTrace(n_post, tau_post)
        self.A_plus = float(A_plus)
        self.A_minus = float(A_minus)
        self.alpha = float(alpha)
        self.w_clip = w_clip

    def decay_to(self, t: float):
        self.pre_trace.decay_to(t)
        self.post_trace.decay_to(t)

    def on_pre_spike(self, i: int, t: float) -> np.ndarray:
        # Decay traces to time t and increment pre trace
        self.decay_to(t)
        self.pre_trace.add(i, 1.0)

        # Learning: potentiation vs Oja decay
        if self.learn:
            y = self.post_trace.val.copy()                 # latency-aware post activity
            oja_term = (y ** 2)[None, :] * self.W[i:i+1, :]
            dW = self.A_plus * (1.0 * y[None, :]) - self.alpha * oja_term
            self.W[i, :] += dW[0]
            np.clip(self.W[i, :], self.w_clip[0], self.w_clip[1], out=self.W[i, :])

        # Current delivered to postsynaptic neurons
        return self.W[i, :].copy()

    def on_post_spike(self, j: int, t: float):
        self.decay_to(t)
        self.post_trace.add(j, 1.0)
        if self.learn:
            x = self.pre_trace.val.copy()
            oja_term = (self.post_trace.val[j] ** 2) * self.W[:, j]
            dW = -self.A_minus * x - self.alpha * oja_term
            self.W[:, j] += dW
            np.clip(self.W[:, j], self.w_clip[0], self.w_clip[1], out=self.W[:, j])

# ---------------------------- Decoder (teacher-driven Oja) ----------------------------
class Decoder:
    """Linear decoder W_out: hidden2 -> outputs, trained with teacher spikes via Oja."""
    def __init__(self, n_pre: int, n_out: int, eta: float = 0.00, alpha: float = 0.001,
                 tau_teacher: float = 20.0, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.1, size=(n_pre, n_out)).astype(np.float64)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.teacher_trace = ExpTrace(n_out, tau_teacher)
        self.pre_trace = ExpTrace(n_pre, tau_teacher)  # driven by hidden spikes
        self.t_last = 0.0

    def decay_to(self, t: float):
        self.teacher_trace.decay_to(t)
        self.pre_trace.decay_to(t)
        self.t_last = t

    def on_pre_spike(self, i: int, t: float) -> np.ndarray:
        self.decay_to(t)
        self.pre_trace.add(i, 1.0)
        # Produce instantaneous output current (for monitoring)
        return self.W[i, :].copy()

    def on_teacher_spike(self, k: int, t: float):
        self.decay_to(t)
        self.teacher_trace.add(k, 1.0)
        # Oja-style supervised update: ΔW[:,k] = η(pre_trace - alpha * y_k^2 * W[:,k])
        yk2 = self.teacher_trace.val[k] ** 2
        dW = self.eta * (self.pre_trace.val - self.alpha * yk2 * self.W[:, k])
        self.W[:, k] += dW

# ---------------------------- Temporal (Latency) Encoder ----------------------------
class LatencyEncoder:
    """Map continuous values in [0, 1] to single spike times within [t0, t0+window].
       Higher value -> earlier spike (shorter latency). Value<=0 -> no spike.
    """
    def __init__(self, window: float = 50.0, t0: float = 0.0, t_eps: float = 1e-3):
        self.window = float(window)
        self.t0 = float(t0)
        self.t_eps = float(t_eps)

    def encode(self, x: np.ndarray) -> List[Tuple[float, int]]:
        """x: shape [N] in [0,1]. Returns list of (time, idx) events."""
        events = []
        for i, v in enumerate(x):
            if v <= 0.0:
                continue
            # earlier time for larger v, clamp to [t0+t_eps, t0+window]
            t = self.t0 + self.t_eps + (1.0 - float(v)) * (self.window - self.t_eps)
            events.append((t, i))
        return events

# ---------------------------- Network ----------------------------
class DeepCausalSNN_Event:
    def __init__(self, n_in: int, n_h1: int, n_h2: int, n_out: int, params: dict):
        # Populations
        # inside DeepCausalSNN_Event.__init__
        self.h1 = LIFPopulation(n=params.get('n_h1', n_h1), tau=params.get('tau', 50.0), v_th=params.get('v_th', .5), inhib=params.get('inhib', 0.05), v_reset=params.get('v_reset', 0.0), refractory=params.get('refractory', 1.0))
        self.h2 = LIFPopulation(n=params.get('n_h2', n_h2), tau=params.get('tau', 50.0), v_th=params.get('v_th', .5), inhib=params.get('inhib', 0.05), v_reset=params.get('v_reset', 0.0), refractory=params.get('refractory', 1.0))

        # Synapses
        self.in_h1 = SynapseMatrix(n_in, n_h1, **params.get('syn_in', {}), name='in->h1')
        self.h1_rec = SynapseMatrix(n_h1, n_h1, **params.get('syn_rec1', {}), name='h1->h1')
        self.h1_h2 = SynapseMatrix(n_h1, n_h2, **params.get('syn_12', {}), name='h1->h2')
        self.h2_rec = SynapseMatrix(n_h2, n_h2, **params.get('syn_rec2', {}), name='h2->h2')
        self.dec = Decoder(n_h2, n_out, **params.get('decoder', {}))

        # Encoders
        self.enc_in = LatencyEncoder(**params.get('enc_in', {}))
        self.enc_out = LatencyEncoder(**params.get('enc_out', {}))

    # --------------- Event processing ---------------
    def run_trial(self, x_vals: np.ndarray, y_vals: np.ndarray,
                  t_max: float = 200.0) -> dict:
        """Run one event-driven trial.
        x_vals: [n_in] in [0,1]
        y_vals: [n_out] in [0,1]
        Returns dict with recorded spikes and simple decoded outputs.
        """
        # Build initial event queue from encoders
        evq: List[Event] = []
        spikes_in = self.enc_in.encode(x_vals)
        for t, i in spikes_in:
            heapq.heappush(evq, Event(t, 'in', i))

        spikes_out = self.enc_out.encode(y_vals)
        for t, k in spikes_out:
            heapq.heappush(evq, Event(t, 'target', k))

        # Records
        spikes_h1: List[Tuple[float, int]] = []
        spikes_h2: List[Tuple[float, int]] = []
        spikes_out_hat: List[Tuple[float, int]] = []  # predicted output spikes (by thresholding accum current)

        # Simple output monitor (not used for learning, just to visualize mapping)
        y_hat = np.zeros(self.dec.W.shape[1], dtype=np.float64)
        y_hat_tlast = 0.0
        tau_yhat = 20.0
        vth_out = 1.0

        def decay_yhat(to_t: float):
            nonlocal y_hat_tlast, y_hat
            dt = max(0.0, to_t - y_hat_tlast)
            if dt > 0.0:
                y_hat *= math.exp(-dt / tau_yhat)
                y_hat_tlast = to_t

        # Process events chronologically
        while evq:
            ev = heapq.heappop(evq)
            # print("event: "+str(ev))
            t = ev.time
            if t > t_max:
                break

            # Decay all states to current time
            self.h1.decay_to(t);
            self.h2.decay_to(t)
            self.in_h1.decay_to(t);
            self.h1_rec.decay_to(t);
            self.h1_h2.decay_to(t);
            self.h2_rec.decay_to(t)
            self.dec.decay_to(t)
            decay_yhat(t)

            if ev.src == 'in':
                # Input spike i -> h1 currents + learning at in->h1
                I = self.in_h1.on_pre_spike(ev.idx, t)       # learn + current vector to h1
                self.h1.receive_current(I)

            elif ev.src == 'h1':
                # h1 spike j -> recurrent to h1 and feedforward to h2
                j = ev.idx
                # STDP post update on h1_rec (post spike at j)
                self.h1_rec.on_post_spike(j, t)
                # Deliver recurrent current (pre spike j)
                Irec = self.h1_rec.on_pre_spike(j, t)
                self.h1.receive_current(Irec)
                # Deliver to layer 2
                I12 = self.h1_h2.on_pre_spike(j, t)
                self.h2.receive_current(I12)

            elif ev.src == 'h2':
                # h2 spike k -> recurrent to h2, and decoder pre
                k = ev.idx
                self.h2_rec.on_post_spike(k, t)
                Irec2 = self.h2_rec.on_pre_spike(k, t)
                self.h2.receive_current(Irec2)
                # Decoder gets pre spike for output monitoring and future teacher learning
                Iout = self.dec.on_pre_spike(k, t)
                y_hat += Iout
                # Threshold predicted output spikes for visualization
                out_spk = np.nonzero(y_hat > vth_out)[0]
                if out_spk.size > 0:
                    for kk in out_spk:
                        spikes_out_hat.append((t, int(kk)))
                        y_hat[kk] = 0.0

            elif ev.src == 'target':
                # Teacher spike for supervised Oja at decoder
                self.dec.on_teacher_spike(ev.idx, t)

            # After delivering currents, check for new spikes in h1/h2 and schedule immediately (no axonal delay)
            new_h1 = self.h1.threshold_spikes(t)
            print(f"t={t:.2f} h1 potentials={self.h1.v} spikes={new_h1}")
            if len(new_h1) > 0:
                print('h1_spike')
                print("new_h1", new_h1)
            # if (self.h1.v > 0.0).any():
                if (self.h1.v > 0.5).any():
                    print("stop")

            if len(new_h1) > 0:
                # self.h1.apply_global_inhibition(len(new_h1))
                for j in new_h1:
                    spikes_h1.append((j[0], int(j[1])))
                    heapq.heappush(evq, Event(j[0], 'h1', int(j[1])))

            new_h2 = self.h2.threshold_spikes(t)
            print(f"t={t:.2f} h2 potentials={self.h2.v} spikes={new_h2}")
            # if (self.h2.v > 1.0).any():
            if len(new_h2) > 0:
                # self.h2.apply_global_inhibition(len(new_h2))
                for k in new_h2:
                    spikes_h2.append((k[0], int(k[1])))
                    heapq.heappush(evq, Event(k[0], 'h2', int(k[1])))

        return {
            'spikes_in': spikes_in,
            'spikes_h1': spikes_h1,
            'spikes_h2': spikes_h2,
            'spikes_out_hat': spikes_out_hat,
            'spikes_out': spikes_out,
            'W_in_h1': self.in_h1.W.copy(),
            'W_h1_h2': self.h1_h2.W.copy(),
            'W_rec1': self.h1_rec.W.copy(),
            'W_rec2': self.h2_rec.W.copy(),
            'W_dec': self.dec.W.copy(),
        }

import matplotlib.pyplot as plt

def plot_spikes_raster(spikes_in, spikes_h1, spikes_h2, spikes_out_hat, spikes_out, t_max):
    """
    Plot spike rasters for input, hidden layers, predicted output, and target spikes.
    Each row = a neuron, x-axis = time, dots = spikes.
    """
    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    # Input
    for t, i in spikes_in:
        axs[0].plot(t, i, 'ko')
    axs[0].set_ylabel("Input")
    axs[0].set_xlim(0, t_max)
    # Hidden 1
    for t, j in spikes_h1:
        axs[1].plot(t, j, 'ro')
    axs[1].set_ylabel("h1")
    # Hidden 2
    for t, k in spikes_h2:
        axs[2].plot(t, k, 'bo')
    axs[2].set_ylabel("h2")
    # Predicted output spikes
    for t, m in spikes_out_hat:
        axs[3].plot(t, m, 'go')
    axs[3].set_ylabel("ŷ")
    # Target spikes (teacher)
    for t, n in spikes_out:
        axs[4].plot(t, n, 'mo')
    axs[4].set_ylabel("Target")
    axs[4].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    np.random.seed(0)
    B, N_IN, N_OUT = 400, 2, 1  # Batch size, input, output dimensions

    params = {
        "n_h1": 8,
        "n_h2": 6,
        "encoding_window": 5}

    # Generate deterministic mapping
    x = np.random.rand(B, N_IN)
    r = (x[:,0]-.5) ** 2 + (x[:,1]-.5) ** 2
    y = np.expand_dims((1 - r) * np.exp(-r /2),1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x[:, 0], x[:, 1], y[:, 0], c=y[:, 0], cmap='viridis', s=50)
    # plt.show(block=True)

    model = DeepCausalSNN_Event(N_IN, params["n_h1"], params["n_h2"], N_OUT, params)

    n_epochs = 100
    losses = []

    for epoch in range(n_epochs):
        print("epoch: "+str(epoch))
        perm = np.random.permutation(B)
        x_shuffled, y_shuffled = x[perm], y[perm]
        y_true_epoch, y_pred_epoch = [], []
        for b in range(B):
            print("sample: "+str(b))
            x_b, y_b = x_shuffled[b], y_shuffled[b]
            trial_out = model.run_trial(x_b, y_b)
            plot_spikes_raster(trial_out['spikes_in'], trial_out["spikes_h1"], trial_out["spikes_h2"], trial_out["spikes_out_hat"], trial_out['spikes_out'], t_max=200)
            y_true_epoch.append(y_b)
            y_hat_b = np.zeros(N_OUT)
            for _, k in trial_out['spikes_out_hat']:
                y_hat_b[k] += 1
            y_pred_epoch.append(y_hat_b)
        y_true_epoch = np.array(y_true_epoch)
        y_pred_epoch = np.array(y_pred_epoch)

        # Mean squared error for this epoch
        mse = np.mean((y_true_epoch - y_pred_epoch) ** 2)
        losses.append(mse)
        print(f"Epoch {epoch+1}/{n_epochs}, MSE: {mse:.4f}")


    # ---- Test loop ----
    # Does it really work?
    y_true_test, y_pred_test = [], []
    for b in range(B):
        trial_out = model.run_trial(x[b], y[b])
        y_true_test.append(y[b])
        y_hat_b = np.zeros(N_OUT)
        for _, k in trial_out['spikes_out_hat']:
            y_hat_b[k] += 1
        y_pred_test.append(y_hat_b)

    y_true_test = np.array(y_true_test)
    y_pred_test = np.array(y_pred_test)

    # ---- Plot test results ----
    fig, axs = plt.subplots(N_OUT, 1, figsize=(8, 4))
    if N_OUT == 1:
        axs = [axs]
    for i in range(N_OUT):
        axs[i].plot(y_true_test[:, i], label="True")
        axs[i].plot(y_pred_test[:, i], label="Pred", marker="o")
        axs[i].set_title(f"Test Output neuron {i}")
        axs[i].legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.show()


    # ---- Plot latency-encoded inputs ----
    enc_in = LatencyEncoder(**params.get('enc_in', {}))
    fig_in, axs_in = plt.subplots(N_IN, 1, figsize=(8, 2 * N_IN), sharex=True)
    if N_IN == 1:
        axs_in = [axs_in]
    for i in range(N_IN):
        for b in range(B):
            axs_in[i].imshow([[x[b, i]]], cmap="Greens", vmin=0, vmax=1,
                             extent=[0, enc_in.window, b - 0.4, b + 0.4],
                             aspect="auto")
            spikes = [t for t, idx in enc_in.encode(x[b]) if idx == i]
            for t in spikes:
                axs_in[i].vlines(t, b - 0.4, b + 0.4, color="black")
        axs_in[i].set_ylabel(f"In {i}")
    axs_in[-1].set_xlabel("Time (ms)")
    fig_in.suptitle("Latency encoding of inputs")
    plt.tight_layout()
    plt.show()

    # ---- Plot latency-encoded targets ----
    enc_out = LatencyEncoder(**params.get('enc_out', {}))
    fig_out, axs_out = plt.subplots(N_OUT, 1, figsize=(8, 2 * N_OUT), sharex=True)
    if N_OUT == 1:
        axs_out = [axs_out]
    for k in range(N_OUT):
        for b in range(B):
            axs_out[k].imshow([[y[b, k]]], cmap="Reds", vmin=0, vmax=1,
                              extent=[0, enc_out.window, b - 0.4, b + 0.4],
                              aspect="auto")
            spikes = [t for t, idx in enc_out.encode(y[b]) if idx == k]
            for t in spikes:
                axs_out[k].vlines(t, b - 0.4, b + 0.4, color="black")
        axs_out[k].set_ylabel(f"Tgt {k}")
    axs_out[-1].set_xlabel("Time (ms)")
    fig_out.suptitle("Latency encoding of targets")
    plt.tight_layout()
    plt.show()






