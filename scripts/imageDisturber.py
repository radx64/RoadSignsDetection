import cv2
import cv
import random

class ImageDisturber:
	def __init__(self, imageArray, noiseFactor):
		self.imageArray = imageArray
		imageSize = self.imageArray.shape[0]
		for pixel in range(0, int(noiseFactor*imageSize)):
			pos = random.randint(0,imageSize-1)
			#print "Mutate " + str(imageSize) + " " + str(pos)
			if self.imageArray[pos] == -1:
				self.imageArray[pos] = 1
				
			else:
				self.imageArray[pos] = -1

	def get(self):
		return self.imageArray