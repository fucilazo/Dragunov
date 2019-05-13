import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Number of particles
N = 100

# Max object movement between frames
STEPSIZE = 10

# Number of frames
N_FRAMES = 40


def resample(weights):
    n = len(weights)
    u0, j = np.random.random(), 0

    C = [0.] + list(np.cumsum(weights))
    indices = []
    for u in [(u0 + i) / n for i in range(n)]:
        while j < len(C) and u > C[j]:
            j += 1
        if j == len(C):
            continue
        indices.append(j - 1)
    return indices


class Particles:

    def __init__(self, initial_square):
        self.positions = np.ones((N, 2), np.uint8) * initial_square[:2]
        self.sizes = np.ones((N, 2), np.uint8) * initial_square[2:]

        # set initial weights to maximum
        self.weights = np.zeros(N, dtype=np.float32)

    def get_oposite_positions(self):
        return self.positions + self.sizes

    def draw_particle(self, image, i):
        cv2.rectangle(image, tuple(np.flip(self.positions[i], 0)), tuple(np.flip(self.positions[i] + self.sizes[i], 0)), (255, 255, 255), thickness=1)


class ParticleFilter:

    HISTOGRAM_BINS = 2

    def __init__(self, ini_frame, initial_square):
        """
        :param ini_frame:
        :param initialSquare: Tuple (y, x, h, w) square to init the tracking
        """
        self.particles = Particles(initial_square)

        h, w, _ = ini_frame.shape
        # Initial samples selections
        # Target colour model
        self.f0 = ini_frame[initial_square[0]:initial_square[0]+initial_square[2], initial_square[1]:initial_square[1] + initial_square[3]] * np.ones((N, 8, 8, 3))
        self.hist0_r = np.histogram(self.f0[:, :, :, 0], bins=self.HISTOGRAM_BINS, range=(.0, 255.))[0].astype(np.float32)
        self.hist0_g = np.histogram(self.f0[:, :, :, 1], bins=self.HISTOGRAM_BINS, range=(.0, 255.))[0].astype(np.float32)
        self.hist0_b = np.histogram(self.f0[:, :, :, 2], bins=self.HISTOGRAM_BINS, range=(.0, 255.))[0].astype(np.float32)

    def process(self, frame):
        # propagate
        np.add(self.particles.positions, np.random.uniform(-STEPSIZE, STEPSIZE, self.particles.positions.shape),
               out=self.particles.positions, casting="unsafe")  # Particle motion model: uniform step

        # Discard out-of-bounds particles
        self.particles.positions = self.particles.positions.clip(np.zeros(2), np.array((im_h, im_w)) - 1).astype(np.int)

        self.__compute_weights(frame)

        # self.position = np.sum(self.particles.T * self.weights, axis=1)  # Return expected position, particles and weights

        # if 1. / np.sum(self.particles.weights ** 2) < N / 2.:  # If particle cloud degenerate:
        new_samples = resample(self.particles.weights)
        if len(new_samples) != 0:
            self.particles.positions = self.particles.positions[new_samples, :]  # Resample particles according to weights

    def __compute_weights(self, frame):
        op_pos = self.particles.get_oposite_positions()
        # Get particle colours
        for i, p in enumerate(self.particles.positions):
            f = frame[p[0]:op_pos[i][0], p[1]:op_pos[i][1]]

            corr_r = cv2.compareHist(np.histogram(f[:, :, 0], bins=self.HISTOGRAM_BINS, range=(.0, 255.))[0].astype(np.float32),
                                     self.hist0_r, method=cv2.HISTCMP_CORREL)
            corr_g = cv2.compareHist(np.histogram(f[:, :, 1], bins=self.HISTOGRAM_BINS, range=(.0, 255.))[0].astype(np.float32),
                                     self.hist0_g, method=cv2.HISTCMP_CORREL)
            corr_b = cv2.compareHist(np.histogram(f[:, :, 2], bins=self.HISTOGRAM_BINS, range=(.0, 255.))[0].astype(np.float32),
                                     self.hist0_b, method=cv2.HISTCMP_CORREL)

            # w = corr_r * 0.1 + corr_b * 0.1 + corr_g * 0.8
            w = corr_g
            self.particles.weights[i] = math.pow(math.e, -16.0 * (1-w))  #  1. / (1. + w ** 2)

        # self.particles.weights /= np.sum(self.particles.weights)


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

    pc = ParticleFilter(im0, (120, 160, 8, 8))

    debug_image = im0.copy()
    for i, _ in enumerate(pc.particles.positions):
        pc.particles.draw_particle(debug_image, i)

    plt.title("test")
    plt.imshow(debug_image)
    plt.pause(0.5)

    for i, im in enumerate(seq):
        pc.process(im)

        debug_image = im.copy()
        for i, _ in enumerate(pc.particles.positions):
            pc.particles.draw_particle(debug_image, i)

        plt.title("test")
        plt.imshow(debug_image)
        plt.pause(0.5)
