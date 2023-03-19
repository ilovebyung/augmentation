'''
Prerequsite size is 224
Run resize.py first 
'''

import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# source = 'D:\\Data\\20210614\\P_resized'
source = 'D:\\Project\\teachable_machine\\N_20210614'

os.chdir(source)

# Positive
for file in os.listdir(source):
    image = Image.open(file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Rotation 90
    rotated_image_90 = image.rotate(90)
    cv2.imwrite(f'90_{file}', np.asarray(rotated_image_90))

    # Rotation 180
    rotated_image_180 = image.rotate(180)
    cv2.imwrite(f'180_{file}', np.asarray(rotated_image_180))

    # Rotation 270
    rotated_image_270 = image.rotate(270)
    cv2.imwrite(f'270_{file}', np.asarray(rotated_image_270))

    # Flipping images
    flipped_image = np.fliplr(image)
    cv2.imwrite(f'f_{file}', np.asarray(flipped_image))

    # Flipping images with rotated_image_90
    flipped_image_90 = np.fliplr(rotated_image_90)
    cv2.imwrite(f'90_f_{file}', np.asarray(flipped_image_90))

    # Flipping images with rotated_image_180
    flipped_image_180 = np.fliplr(rotated_image_180)
    cv2.imwrite(f'180_f_{file}', np.asarray(flipped_image_180))

    # Flipping images with rotated_image_270
    flipped_image_270 = np.fliplr(rotated_image_270)
    cv2.imwrite(f'270_f_{file}', np.asarray(flipped_image_270))
