import random
import numpy as np
import cv2
import cv

class ImgPreprocessor:
	def __init__(self, imageName, size, tresholdLevel=127):
		self.imageName = imageName
		self.size = size
		self.tresholdLevel = tresholdLevel
		self.imageData = self.openImage(self.imageName)
		self.imageDataHopfield = np.zeros((self.size, self.size))
		self.resizeImage()
		self.tresholdImage()
		self.convertToOnes()

	def openImage(self, imageName):
		return cv2.imread(imageName,0)

	def tresholdImage(self):
		ret, self.imageData = cv2.threshold(self.imageData,self.tresholdLevel,255,cv2.THRESH_BINARY)

	def resizeImage(self):
		self.imageData = cv2.resize(self.imageData, (self.size,self.size));

	def convertToOnes(self):
		for i, line in enumerate(self.imageData):
			for j, element in enumerate(line):
				if element == 255:
					result = 1
				else:
					result = -1
				self.imageDataHopfield[i,j] = result

	def getImage(self):
		return self.imageData

	def getImageForHopfield(self):
		return self.imageDataHopfield.ravel()