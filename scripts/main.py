#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

from hopfield import HopfieldNetwork
from imgPreprocessor import ImgPreprocessor

trainSet = np.array(
		[
			[-1,-1, 1, 1,-1,-1],
			[ 1, 1,-1,-1, 1, 1]
		])

def main():
	imgProc = ImgPreprocessor('../img/b_33.png',12,127)
	image = imgProc.getImage();
	imageForHopfield = imgProc.getImageForHopfield()
	print imageForHopfield

	net = HopfieldNetwork(imageForHopfield.shape[0])
	net.initWeights(imageForHopfield.shape[0])
	net.trainHebb(imageForHopfield)
	net.initNeurons(np.ones(imageForHopfield.shape[0]))

	net.update(1000)
	print "Results after 1000 updates:"
	remeberedImage = net.getNeuronsMatrix()
	print remeberedImage

	remeberedImage = np.reshape(remeberedImage, (image.shape[0],image.shape[0]))

	plt.subplot(121),plt.imshow(image,cmap = 'gray', interpolation = 'nearest')
	plt.title('Thresholded Image'), plt.xticks([]), plt.yticks([])	

	plt.subplot(122),plt.imshow(remeberedImage,cmap = 'gray', interpolation = 'nearest')
	plt.title('Thresholded Image'), plt.xticks([]), plt.yticks([])
	plt.show()

if __name__ == '__main__':
	main()