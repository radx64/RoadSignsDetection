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

	net.update(100)
	remeberedImage100 = net.getNeuronsMatrix().copy()
	remeberedImage100 = np.reshape(remeberedImage100, (image.shape[0],image.shape[0]))
	net.update(200)
	remeberedImage300 = net.getNeuronsMatrix().copy()
	remeberedImage300 = np.reshape(remeberedImage300, (image.shape[0],image.shape[0]))
	net.update(300)
	remeberedImage600 = net.getNeuronsMatrix().copy()
	remeberedImage600 = np.reshape(remeberedImage600, (image.shape[0],image.shape[0]))
	net.update(400)
	remeberedImage1000 = net.getNeuronsMatrix().copy()
	remeberedImage1000 = np.reshape(remeberedImage1000, (image.shape[0],image.shape[0]))

	plt.subplot(241),plt.imshow(image,cmap = 'gray', interpolation = 'nearest')
	plt.title('Orginal image'), plt.xticks([]), plt.yticks([])	

	plt.subplot(245),plt.imshow(remeberedImage100,cmap = 'gray', interpolation = 'nearest')
	plt.title('Image after 100 iterations'), plt.xticks([]), plt.yticks([])
	plt.subplot(246),plt.imshow(remeberedImage300,cmap = 'gray', interpolation = 'nearest')
	plt.title('Image after 300 iterations'), plt.xticks([]), plt.yticks([])
	plt.subplot(247),plt.imshow(remeberedImage600,cmap = 'gray', interpolation = 'nearest')
	plt.title('Image after 600 iterations'), plt.xticks([]), plt.yticks([])
	plt.subplot(248),plt.imshow(remeberedImage1000,cmap = 'gray', interpolation = 'nearest')
	plt.title('Image after 1000 iterations'), plt.xticks([]), plt.yticks([])
	plt.show()

if __name__ == '__main__':
	main()