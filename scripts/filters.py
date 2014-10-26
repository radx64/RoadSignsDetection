#!/usr/bin/python
import sys
import random
import numpy as np
import cv2
import cv

import matplotlib.pyplot as plt

def redFilter(inputImageInHSV):
	redFilterDown = np.array([0,50,50]);
	redFilterUp = np.array([20,255,255]);	
	return cv2.inRange(inputImageInHSV,redFilterDown,redFilterUp)

def blueFilter(inputImageInHSV):
	blueFilterDown = np.array([50,50,50]);
	blueFilterUp = np.array([120,255,255]);
	return cv2.inRange(inputImageInHSV,blueFilterDown,blueFilterUp)

def yellowFilter(inputImageInHSV):
	yellowFilterDown = np.array([25,50,50]);
	yellowFilterUp = np.array([35,255,255]);
	return cv2.inRange(inputImageInHSV,yellowFilterDown,yellowFilterUp) 

imageName = sys.argv[1]

print "Loading image " + imageName + " in color"

image = cv2.imread(imageName)
RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

redFilterImg = redFilter(HSV);
blueFilterImg = blueFilter(HSV);
yellowFilterImg = yellowFilter(HSV)

#cv2.imshow('Orginal image', image)
#cv2.imshow('Red only', redFilterImg)
#cv2.imshow('Blue only', blueFilterImg)
#cv2.imshow('Yellow only', yellowFilterImg)

print "Loading image " + imageName + " in grayscale"
grayscale = cv2.imread(imageName,0)

edgeImg = cv2.Canny(grayscale,100,200);

plt.subplot(231),plt.imshow(RGB)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(edgeImg,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(redFilterImg, cmap = 'gray')
plt.title('Red filtered'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(blueFilterImg, cmap = 'gray')
plt.title('Blue filtered'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(yellowFilterImg, cmap = 'gray')
plt.title('Yellow filtered'), plt.xticks([]), plt.yticks([])



plt.show()

#cv2.imshow('Edged image',edgeImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
