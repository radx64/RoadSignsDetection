import cv2
import cv

class ImageSaver:
	def __init__(self, imageName, imageArray, side):
		cv2.imwrite(imageName, self.convertToImg(imageArray).reshape(side,side))

	def convertToImg(self, matrix):
		matrix[matrix == -1] = 0
		matrix[matrix == 1] = 255
		return matrix