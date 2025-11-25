import numpy as np


def latency_encode(x, t_min=0.0, t_max=10.0):
    """
    Method 1: encode magnitude as spike latency.
    Input x must be in range [0, 1].
    Higher x → earlier spike.
    """
    return t_min + (1 - x) * (t_max - t_min)


def dendritic_matcher(spike_times, delays, tau=3.0):
    """
    Simple dendritic matcher:
    - Each synapse has a presynaptic spike at time 'spike_times[i]'
    - Each synapse expects an arrival at 'delays[i]'
    - We compute mismatch: abs((spike_times + delays) - aligned_time)
    - If mismatches align well, post spike occurs early
    """
    # Compute "arrival times" at soma
    arrival_times = spike_times + delays

    # Good match = low variance across arrival times
    variance = np.var(arrival_times)

    # Map variance to output spike latency
    # Perfect match (variance=0) → early spike
    # Bad match → late spike
    t_out = 5.0 + variance  # just for illustration

    return t_out, arrival_times


# ----- Example -----

# A short sliding window of input magnitudes
x = np.array([0.1, 0.3, 0.5, 0.7])  # raw input subsequence

# Step 1: Convert values into spike latencies
spike_times = latency_encode(x)

# Step 2: Fixed dendritic delays encoding a motif
delays = np.array([0.0, 2.0, 4.0, 6.0])  # the "pattern" the dendrite expects

# Step 3: Run matcher
t_out, arrival_times = dendritic_matcher(spike_times, delays)

print("Input values:", x)
print("Encoded spike times:", spike_times)
print("Dendritic delays:", delays)
print("Arrival times:", arrival_times)
print("Output spike time (matcher):", t_out)