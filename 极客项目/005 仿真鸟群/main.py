import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist
from numpy.linalg import norm

width, height = 640, 480


class Boids:
    def __init__(self, N):
        self.pos = [width/2.0, ]

    def tick(self, frameNum, pts, beak):
        pass

    def limitVec(self, vec, maxVal):
        pass

    def limit(self, X, maxVal):
        pass

    def applyBC(self):
        pass

    def applyRules(self):
        pass

    def buttonPress(self, event):
        pass


def tick(frameNum, pts, beak, boids):
    pass


def main():
    pass


if __name__ == '__main__':
    main()
