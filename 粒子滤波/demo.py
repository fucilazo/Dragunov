'''
1D PARTICLE FILTER
Author: Kaushik Balakrishnan, PhD
Email: kaushikb258@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import sys


def get_xtp1(t):
    vt = 0.9 * np.random.randn()
    xtp1 = 50.0 + 1.6 * np.sin(3.5 * np.pi * t) - 5.3 * t + vt
    return xtp1


def get_yt(xt, t):
    nt = 1.4 * np.random.randn()
    yt = xt + nt
    return yt


def calc_weights(Np, x, y, w):
    sigma = 2.0
    for i in range(Np):
        w[i] = 1.0 / 2.0 / np.pi / sigma * np.exp(-(x[i] - y) * (x[i] - y) / 2.0 / sigma / sigma)
        if (w[i] > 0.1):
            w[i] = 0.1  # don't allow weights to become too large
    w = w / np.sum(w)
    w = np.array(w)
    return w


def sample(Np, w, x):
    x1 = np.zeros((Np), dtype=np.float)
    cw = np.zeros((Np), dtype=np.float)

    cw[0] = w[0]
    for i in range(1, Np):
        cw[i] = cw[i - 1] + w[i]

    for i in range(Np):
        r = np.random.rand()
        j = 0
        while (cw[j] < r):
            j += 1
        x1[i] = x[j] + 0.2 * np.random.randn()  # add noise to avoid overlap of particles

    # randomly set 10% of the particles to explore new regions
    for i in range(int(0.1 * Np)):
        j = np.random.randint(0, Np - 1)
        x1[j] = xmax * np.random.rand()

    return x1


# --------------------------------------------------------------------

x0 = 50.0  # initial location
xmax = 100.0  # max x location
dt = 0.01  # time step

nt = 500  # numbeer of time steps
x = np.zeros((nt), dtype=np.float)
tt = np.zeros((nt), dtype=np.float)
y = np.zeros((nt), dtype=np.float)
x[0] = x0

Np = 100  # numper pf particles
xpart = np.zeros((Np), dtype=np.float)
weights = np.zeros((Np), dtype=np.float)

# initialize particle locations and weights 
for i in range(Np):
    xpart[i] = xmax * np.random.rand()

ip = np.zeros((Np), dtype=np.float)

# --------------------------------------------------------------------

imgno = 0

# take measurements
for i in range(nt):
    print("step numbeer: ", i)
    tt[i] = dt * np.float(i)
    if (i < nt - 1):
        x[i + 1] = get_xtp1(tt[i])
    y[i] = get_yt(x[i], tt[i])
    weights = calc_weights(Np, xpart, y[i], weights)
    xpart = sample(Np, weights, xpart)
    for k in range(Np):
        ip[k] = tt[i]

    print("max/min weight: ", np.max(weights), np.min(weights))

    plt.ion()
    plt.plot(x[:i], tt[:i], c='r', label='truth x')
    plt.plot(y[:i], tt[:i], 'o', c='g', label='measured x', markersize=5)
    plt.legend(fontsize=15, numpoints=1)
    plt.scatter(xpart, ip, s=5)  # weights*10)
    plt.axis([0, xmax, 0, dt * nt])
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.ylabel("time", fontsize=20)
    plt.xlabel("x-location", fontsize=20)
    plt.title("1D particle filter", fontsize=20)
    # plt.show()
    # plt.pause(0.1)
    # if (i % 100 == 0 or i == nt - 1):
    #     plt.savefig("particle_filter_" + str(imgno) + ".png")
    #     imgno += 1
    # plt.clf()
    plt.show()