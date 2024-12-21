## File: /real-time-heart-rate-detection/real-time-heart-rate-detection/src/utils/heart_rate.py

import numpy as np
import cv2
import scipy.signal as signal

def cpu_POS(signal, fps):
    eps = 10**-9
    X = signal
    e, c, f = X.shape
    w = int(1.6 * fps)

    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    H = np.zeros((e, f))
    for n in np.arange(w, f):
        m = n - w + 1
        Cn = X[:, :, m:(n+1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)
        Cn = np.multiply(Cn, M)

        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)

        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)

        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H
