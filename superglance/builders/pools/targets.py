"""Imitation soft targets: softmax(-ADE / tau) over the K candidates, ADE = mean distance over the 8 poses at 0.5-4 s
(tau 1 m for navtrain / OpenScene and WOD rows, 0.5 m for PAI-AV rows); float32 rows computed in float64."""
import json, os, shutil, sys
import numpy as np


def soft(F, C, tau):   # [N, 8, 2] -> [N, K] float32 softmax(-ADE / tau), computed in float64
    out = np.empty((len(F), len(C)), np.float32)
    for i in range(0, len(F), 256):
        a = np.linalg.norm(C[None] - F[i:i + 256, None], axis=-1).mean(-1)
        q = np.exp(-(a - a.min(1, keepdims=True)) / tau); out[i:i + 256] = q / q.sum(1, keepdims=True)
    return out


def selftest():   # soft() is a distribution peaked at the nearest candidate, and the tau sets its sharpness
    C = np.array([np.c_[np.arange(1, 9) * v, np.zeros(8)] for v in (1.0, 2.0, 3.0)])
    q, q2 = soft(C[1:2] + 0.1, C, 0.5)[0], soft(C[1:2] + 0.1, C, 2.0)[0]
    assert q.argmax() == 1 and abs(q.sum() - 1) < 1e-6 and q[1] > q2[1] > 1 / 3


if __name__ == "__main__":
    selftest()
