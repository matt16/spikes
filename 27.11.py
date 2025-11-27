import numpy as np

# ------------------------------------------------------------
# Exponential synaptic kernel: converts spike times -> current
# I(t) = w * exp(-(t - t_spike)/tau)  for t > t_spike
# ------------------------------------------------------------
def syn_current(t, spikes, w, tau):
    I = 0.0
    for ts in spikes:
        if t >= ts:
            I += w * np.exp(-(t - ts) / tau)
    return I


# ------------------------------------------------------------
# Dendritic branch: integrates synaptic current into voltage
# Using simple LIF passive integration (no spiking here)
# ------------------------------------------------------------
class DendriteBranch:
    def __init__(self, weights, tau_syn=5.0, tau_mem=20.0):
        self.weights = weights
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem

    def run(self, spike_inputs, t_max=50.0, dt=0.1):
        """
        spike_inputs = [ [t1, t2, ...], [t1, t2, ...], ... ]
        """
        t_values = np.arange(0, t_max, dt)
        v = 0.0
        vs = []

        for t in t_values:
            # 1) compute synaptic current
            I = 0.0
            for spikes, w in zip(spike_inputs, self.weights):
                I += syn_current(t, spikes, w, self.tau_syn)

            # 2) membrane integration: dv = (-v + R*I) * dt/tau_mem
            dv = (-v + I) * (dt / self.tau_mem)
            v += dv
            vs.append(v)

        # 3) convert max voltage -> latency spike (latency coding)
        spike_t = t_values[np.argmax(vs)]
        return spike_t, np.array(vs), t_values


# ------------------------------------------------------------
# Multi-branch hierarchical dendrite feeding a main LIF soma
# ------------------------------------------------------------
class HierarchicalDendriticUnit:
    def __init__(self, level1_weights, level2_weights):
        # Level 1 branches (parallel)
        self.branches_L1 = [DendriteBranch(w) for w in level1_weights]

        # Level 2 branch (collects from L1)
        self.branch_L2 = DendriteBranch(level2_weights)

    def run(self, spike_inputs_L1):
        """
        spike_inputs_L1 = list of list-of-spike-times for each L1 branch
        """
        # -----------------------------
        # Run Level 1 branches
        # -----------------------------
        L1_spike_times = []
        for branch, spikes in zip(self.branches_L1, spike_inputs_L1):
            t_spike, _, _ = branch.run([spikes])
            L1_spike_times.append([t_spike])

        # -----------------------------
        # Level 2 branch takes *times from L1*
        # -----------------------------
        L2_spike_time, v_trace, t_axis = self.branch_L2.run(L1_spike_times)

        return L2_spike_time, v_trace, t_axis


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
# three input branches in level 1
level1_weights = [
    [1.2],
    [0.7],
    [1.0],
]

# level 2 receives 3 spike times from level 1 branches
level2_weights = [1.0, 0.9, 1.1]

unit = HierarchicalDendriticUnit(level1_weights, level2_weights)

# Input spike times for each L1 branch
input_spikes = [
    [10.0],   # branch 1
    [15.0],   # branch 2
    [20.0],   # branch 3
]

out_spike, v_trace, t_axis = unit.run(input_spikes)
print("Output spike time:", out_spike)