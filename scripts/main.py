#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import sys

from hopfield import HopfieldNetwork
from imgPreprocessor import ImgPreprocessor

def loadParameters(parametersArray):
	if(len(parametersArray) < 3):
		print "Need to give more parameters"
		sys.exit(1)
	signs = list()
	imageToRecognize = ""
	print "Number of signs to learn: " + str(len(parametersArray)-1)
	for idx, sign in enumerate(parametersArray):
		if (idx > 0) and (idx<len(parametersArray)-1):
			signs.append(sign)
			print "Image to learn: " + str(sign)
	print "Image to recognize: " + parametersArray[-1]
	imageToRecognize = parametersArray[-1]
	return signs, imageToRecognize

def main():
	SIDE_OF_ARRAY = 12

	net = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)

	signsList, imgToRecogn = loadParameters(sys.argv)
	images = list()
	for image in signsList:
		imgProc = ImgPreprocessor(image,SIDE_OF_ARRAY,127)
		image = imgProc.getImageForHopfield().copy();
		print image
		images.append(image)
		net.trainHebb(image)

	imgProc = ImgPreprocessor(imgToRecogn,SIDE_OF_ARRAY,127)
	imageRec = imgProc.getImageForHopfield();

	net.initNeurons(imageRec);
	net.update(2000);

	subplotIndex = 201
	subplotIndex += 10 * len(images)
	
	for idx, image in enumerate(images):
		plt.subplot(subplotIndex+idx)
		image2d = np.reshape(image, (SIDE_OF_ARRAY,SIDE_OF_ARRAY))
		plt.imshow(image2d, cmap='gray', interpolation = 'nearest')
		plt.xticks([]), plt.yticks([]), plt.title("Sample %d" % idx)

	newLineSubplotIndex = subplotIndex + len(images)
	plt.subplot(newLineSubplotIndex)
	image2d = np.reshape(imageRec, (SIDE_OF_ARRAY,SIDE_OF_ARRAY))
	plt.imshow(image2d, cmap='gray', interpolation = 'nearest')
	plt.xticks([]), plt.yticks([]), plt.title("To recognize")

	plt.subplot(newLineSubplotIndex+1)
	networkResult = net.getNeuronsMatrix().copy()

	image2dResult = np.reshape(networkResult, (SIDE_OF_ARRAY,SIDE_OF_ARRAY))
	print image2dResult
	plt.imshow(image2dResult, cmap='gray', interpolation = 'nearest')
	plt.xticks([]), plt.yticks([]), plt.title("Returned by net")
	plt.show()

if __name__ == '__main__':
	main()