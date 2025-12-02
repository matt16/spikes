import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
torch.set_printoptions(precision=4, sci_mode=False)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




def hankel_windows_multichannel(x, T, stride=1):
    channels_, L = x.shape
    num_windows = (L - T) // stride + 1
    out = np.zeros((num_windows, channels_, T), dtype=x.dtype)
    idx = 0
    for i in range(0, L - T + 1, stride):
        out[idx] = x[:, i:i+T]
        idx += 1
    return out



def magnitude_to_latency_batch(windows, t_min=0.0, t_max=40.0, eps=1e-8):
    nw, ch, T_ = windows.shape
    lat = np.zeros_like(windows, dtype=float)
    for c in range(ch):
        vals = windows[:, c, :].reshape(-1)
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < eps:
            vmax = vmin + eps
        norm = (windows[:, c, :] - vmin) / (vmax - vmin)
        lat[:, c, :] = t_min + (1.0 - norm) * (t_max - t_min)
    return lat




if __name__ == '__main__':
    # -------------------- Config (smaller) --------------------
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    channels = 3
    T = 16
    seq_len = 300   # smaller
    stride = 1
    K = 6   # matchers per channel (mini-neurons)
    t_min, t_max = 0.0, 40.0
    df = pd.read_excel('returns.xlsx')
    series = df.values()

    windows = hankel_windows_multichannel(series, T, stride=stride)  # (num_windows, channels, T)
    num_windows = windows.shape[0]
    latencies = magnitude_to_latency_batch(windows, t_min=t_min, t_max=t_max)  # (num_windows, channels, T)
    split = int(0.8 * num_windows)
    train_lat = torch.from_numpy(latencies[:split]).float()
    val_lat = torch.from_numpy(latencies[split:]).float()

