import cv2
import cv
import random

class ImageLines:
	def __init__(self, imageArray, freq):
		self.imageArray = imageArray
		imageSize = self.imageArray.shape[0]
		for pixel in range(0, imageSize, int(freq)):
			if self.imageArray[pixel] == -1:
				self.imageArray[pixel] = 1
				
			else:
				self.imageArray[pixel] = -1

	def get(self):
		return self.imageArray