#!/usr/bin/python
import sys
import random
import numpy as np
import cv2
import cv

import matplotlib.pyplot as plt

imageName = sys.argv[1]

SIDE_SIZE = 12

print "Loading image " + imageName + " in gray"
image = cv2.imread(imageName,0)

resizedImg = cv2.resize(image, (SIDE_SIZE,SIDE_SIZE));

plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(resizedImg,cmap = 'gray',interpolation="nearest")
plt.title('Scaled Image'), plt.xticks([]), plt.yticks([])

plt.show()