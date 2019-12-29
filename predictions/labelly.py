import cv2
from random import randint
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('pred_5.png')

colors = []

for i in range(255):
    colors.append((randint(0,255), randint(0,255), randint(0,255)))

for h in range(img.shape[0]):
    for w in range(img.shape[1]):
        if len(img.shape) == 3 and img.shape[2] == 3:
            b, g, r = img[h, w]
            key = b
        else:
            key = img[h, w]

        img[h, w] = colors[key]

cv2.imwrite('image_pred_labelled.png', img)
