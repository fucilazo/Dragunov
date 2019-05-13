import glob
import sys
import cv2  # imread
import torch
import numpy as np
import os
from os.path import realpath, dirname, join

image_files = sorted(glob.glob('C:\Dragunov/Tracking/PF-DaSRPN/CarScale/*.jpg'))

if not image_files:
    sys.exit(0)

im = cv2.imread(image_files[0])

for f, image_file in enumerate(image_files):
    if not image_file:
        break
    im = cv2.imread(image_file)

    cv2.imshow('canny3', cv2.Canny(im, 120, 300))
    cv2.waitKey(50)     # 50ms


