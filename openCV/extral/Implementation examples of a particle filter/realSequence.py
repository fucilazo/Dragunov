import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


##################################
##################################
# PARTICLE FILTER DEFINITION     #
##################################
##################################

# Number of particles
N = 200

# Max object movement between frames
STEP_SIZE_POS = 25
STEP_SIZE_SIZE = 8

MAX_SQUARE_W = 50
MAX_SQUARE_H = 110

MIN_SQUARE_W = 30
MIN_SQUARE_H = 80

USE_HSV = False

SHOWN_FREQ = 1/25

# def resample(weights):
#     return resampling.systematic_resample(weights)

def resample(weights):
    n = len(weights)
    u0, j = np.random.random(), 0

    C = [0.] + list(np.cumsum(weights))
    indices = []
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


class Particles:

    def __init__(self, initial_square):
        self.positions = np.ones((N, 2), np.uint8) * initial_square[:2]
        self.sizes = np.ones((N, 2), np.uint8) * initial_square[2:]

        # set initial weights to maximum
        self.weights = np.zeros(N, dtype=np.float64)

    def get_oposite_positions(self):
        min_len = np.min([len(self.sizes), len(self.positions)])
        return self.positions[:min_len] + self.sizes[:min_len]

    def draw_particle(self, image, i):
        cv2.rectangle(image, tuple(np.flip(self.positions[i], 0)), tuple(np.flip(self.positions[i] + self.sizes[i], 0)), (255, 255, 255), thickness=1)


class ParticleFilter:

    HISTOGRAM_BINS = (8, 8, 4) if USE_HSV else (8, 8, 8)

    HIST_COMP = cv2.HISTCMP_BHATTACHARYYA

    def __init__(self, ini_frame, initial_square):
        """
        :param ini_frame:
        :param initial_square: Tuple (y, x, h, w) square to init the tracking
        """
        self.particles = Particles(initial_square)

        self.im_h, self.im_w, _ = ini_frame.shape
        # Initial samples selections
        # Target colour model
        self.f0 = ini_frame[initial_square[0]:initial_square[0]+initial_square[2],
                            initial_square[1]:initial_square[1] + initial_square[3]]
        self.hist0_r = self.get_hist(self.f0, 0)
        self.hist0_g = self.get_hist(self.f0, 1)
        self.hist0_b = self.get_hist(self.f0, 2)

        self.selected = (
            self.particles.positions[0],
            self.particles.sizes[0]
        )

    def process(self, frame):

        self._propagate()

        self.__compute_weights(frame)

        self._select_best_particle(frame)

        # if 1. / np.sum(self.particles.weights ** 2) < N / 2.:  # If particle cloud degenerate:
        new_samples = resample(self.particles.weights)
        print("resampled:")
        print(sorted(new_samples))
        # Resample particles according to weights
        self.particles.positions = self.particles.positions[new_samples, :]
        self.particles.sizes = self.particles.sizes[new_samples, :]  # Resample particles according to weights

    def _propagate(self):

        # propagate
        np.add(self.particles.positions,
               np.random.uniform(-STEP_SIZE_POS, STEP_SIZE_POS, self.particles.positions.shape),
               out=self.particles.positions, casting="unsafe")  # Particle motion model: uniform step

        np.add(self.particles.sizes, np.random.normal(-STEP_SIZE_SIZE, STEP_SIZE_SIZE, self.particles.sizes.shape),
               out=self.particles.sizes, casting="unsafe")

        # Discard out-of-bounds particles
        self.particles.positions = self.particles.positions.clip(np.zeros(2),
                                                                 np.array((self.im_h, self.im_w)) - 1).astype(np.int)
        self.particles.sizes = self.particles.sizes.clip(np.array((MIN_SQUARE_H, MIN_SQUARE_W)),
                                                         np.array((MAX_SQUARE_H, MAX_SQUARE_W)) - 1).astype(np.int)

    def _select_best_particle(self, frame):
        idx_best = np.argmax(self.particles.weights)
        self.selected = (
            self.particles.positions[idx_best],
            self.particles.sizes[idx_best]
        )

        # self.f0 = frame[self.selected[0][0]:self.selected[0][0] + self.selected[1][0],
        #           self.selected[0][1]:self.selected[0][1] + self.selected[1][1]]
        # self.hist0_r = self.get_hist(self.f0, 0)
        # self.hist0_g = self.get_hist(self.f0, 1)
        # self.hist0_b = self.get_hist(self.f0, 2)

    def __compute_weights(self, frame):
        op_pos = self.particles.get_oposite_positions()
        # Get particle colours
        for i, p in enumerate(self.particles.positions):
            f = frame[p[0]:op_pos[i][0], p[1]:op_pos[i][1]]


            self.particles.weights[i] = self.particle_dist(f)

        self.particles.weights = np.power(math.e, -16.0 * (1-self.particles.weights))  # 1. / (1. + w ** 2)
        # self.particles.weights /= np.sum(self.particles.weights)

    def particle_dist(self, frame_part):

        corr_r = cv2.compareHist(
            self.get_hist(frame_part, 0),
            self.hist0_r, method=self.HIST_COMP)
        corr_g = cv2.compareHist(
            self.get_hist(frame_part, 1),
            self.hist0_g, method=self.HIST_COMP)
        corr_b = cv2.compareHist(
            self.get_hist(frame_part, 2),
            self.hist0_b, method=self.HIST_COMP)

        w = corr_r * 0.4 + corr_b * 0.3 + corr_g * 0.3
        # w = corr_r
        return 1 - w

    def get_hist(self, im, n_channel):
        return np.histogram(im[:, :, n_channel], bins=self.HISTOGRAM_BINS[n_channel],
                            range=(.0, 255.))[0].astype(np.float32)


##################################
##################################
#      VISUALIZATION LOGIC       #
##################################
##################################


seq = sorted(glob("/home/bernat/datasets/vot2015/basketball/*.jpg"))
im0 = cv2.cvtColor(cv2.imread(seq[0]), cv2.COLOR_BGR2RGB)
seq = seq[1:]

pc = ParticleFilter(im0, (220, 330, 100, 50)) # config basket
im = im0 if not USE_HSV else cv2.cvtColor(im0, cv2.COLOR_RGB2HSV)

hist_r = pc.get_hist(pc.f0, 0)
hist_g = pc.get_hist(pc.f0, 1)
hist_b = pc.get_hist(pc.f0, 2)

hist_r /= np.sum(hist_r)
hist_g /= np.sum(hist_g)
hist_b /= np.sum(hist_b)


def map_prob(r, g, b):
    return hist_r[int(r)], hist_g[int(g)], hist_b[int(b)]


prob_trans = np.vectorize(map_prob)


def get_debug_particles(im, pc):
    # build particles image
    debug_image = cv2.cvtColor(im.copy(), cv2.COLOR_HSV2RGB) if USE_HSV else im.copy()
    # for i, _ in enumerate(pc.particles.positions):
    #     pc.particles.draw_particle(debug_image, i)

    cv2.rectangle(debug_image, tuple(np.flip(pc.selected[0], 0)),
                  tuple(np.flip(pc.selected[0] + pc.selected[1], 0)),
                  (255, 255, 0), thickness=3)

    # build squares proba image
    im_h, im_w, _ = im.shape
    squares_proba = np.zeros((im_h, im_w), dtype=np.float32)

    x_positions = np.arange(0, im_w, 15)
    y_positions = np.arange(0, im_h, 15)

    y_idxs, x_idxs = np.meshgrid(y_positions, x_positions)
    for i in range(len(y_idxs)):
        for j in range(len(y_idxs[i])):
            y, x = y_idxs[i][j], x_idxs[i][j]
            im_sq = im[y:y + 15, x:x + 15]

            squares_proba[y:y + 15, x:x + 15] = pc.particle_dist(im_sq)
    squares_proba[0, 0] = 0
    squares_proba[0, 1] = 1

    return debug_image, squares_proba


debug_image, proba_im = get_debug_particles(im0, pc)

fig, axes = plt.subplots(1, 2)
axes = axes.flatten()
axes[0].imshow(debug_image)
axes[1].imshow(proba_im)
plt.pause(SHOWN_FREQ)

for i, im_path in enumerate(seq):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    im = im if not USE_HSV else cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    pc.process(im)

    debug_image, proba_im = get_debug_particles(im, pc)
    axes[0].imshow(debug_image)
    axes[1].imshow(proba_im)
    plt.pause(SHOWN_FREQ)
