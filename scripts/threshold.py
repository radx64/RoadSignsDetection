#!/usr/bin/python
import sys
import random
import numpy as np
import cv2
import cv

import matplotlib.pyplot as plt

imageName = sys.argv[1]

print "Loading image " + imageName + " in gray"
image = cv2.imread(imageName,0)

ret, treshImg = cv2.threshold(image,127,255,cv2.THRESH_BINARY);

print "Threshold has returned " + str(ret)

plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(treshImg,cmap = 'gray')
plt.title('Thresholded Image'), plt.xticks([]), plt.yticks([])

plt.show()