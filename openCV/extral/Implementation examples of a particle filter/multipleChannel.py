import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Number of particles
N = 100

# Max object movement between frames
STEPSIZE = 8


def resample(weights):
    n = len(weights)
    u0, j = np.random.random(), 0

    C = [0.] + [sum(weights[:i + 1]) for i in range(n)]
    indices = []
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


class ParticleFilter:
    def __init__(self, firstFrame, initialPos):
        self.position = initialPos

        # Initial samples selections
        self.particles = np.ones((N, 2), np.int) * self.position
        self.f0 = (firstFrame[self.position]) * np.ones((N, 3))  # Target colour model

        # set initial weights to maximum
        self.weights = np.ones((N, 3)) / N

    def process(self, frame):
        # propagate
        np.add(self.particles, np.random.uniform(-STEPSIZE, STEPSIZE, self.particles.shape), out=self.particles,
               casting="unsafe")  # Particle motion model: uniform step

        # Discard out-of-bounds particles
        self.particles = self.particles.clip(np.zeros(2), np.array((im_h, im_w)) - 1).astype(np.int)

        self.weights = self.__compute_weights(frame)

        # if 1. / np.sum(self.weights ** 2) < N / 2.:  # If particle cloud degenerate:
        self.particles = self.particles[resample(self.weights), :]  # Resample particles according to weights

    def __compute_weights(self, frame):
        # Get particle colours
        f = frame[tuple(self.particles.T)]
        diff = self.f0 - f
        ch_r = diff[:, 0]
        ch_g = diff[:, 1]
        ch_b = diff[:, 2]
        w = ch_r * 0.1 + ch_b * 0.1 + ch_g * 0.8
        w = 1. / (1. + w ** 2)
        w /= np.sum(w)
        return w


def generate_sequence(x0, im_h, im_w):
    seq = [im for im in np.zeros((30, im_h, im_w, 3), np.uint8)]

    # and Add a square with starting position x0 moving along trajectory xs
    xs = np.vstack((np.arange(30) * 3, np.arange(30) * 2)).T + x0
    for t, x in enumerate(xs):
        seq[t][x[0]:x[0] + 8, x[1]:x[1] + 8] = (0, 255, 0)

        # draw some disturbing objects
        cv2.line(seq[t], (0, 180), (im_w, 180), (255, 0, 0), thickness=5)
        cv2.line(seq[t], (0, 80), (im_w, 190), (0, 127, 127), thickness=5)

    return seq


if __name__ == "__main__":

    # define the size of the images
    im_h, im_w = 240, 320

    # Create an image sequence of 20 frames long
    x0 = np.array([120, 160])
    seq = generate_sequence(x0, im_h, im_w)
    im0 = seq[0]
    seq = seq[1:]

    pc = ParticleFilter(im0, tuple(x0))

    debug_image = im0.copy()
    debug_image[pc.particles[:, 0], pc.particles[:, 1]] = (0, 0, 255)

    plt.title("test")
    plt.imshow(debug_image)
    plt.pause(0.5)

    for i, im in enumerate(seq):
        pc.process(im)

        debug_image = im.copy()
        debug_image[pc.particles[:, 0], pc.particles[:, 1]] = (255, 255, 255)

        plt.title("test")
        plt.imshow(debug_image)
        plt.pause(0.5)
