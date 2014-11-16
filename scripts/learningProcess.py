#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import sys

from hopfield import HopfieldNetwork
from imgPreprocessor import ImgPreprocessor

def main():
	imgProc = ImgPreprocessor(sys.argv[1],12,127)
	image = imgProc.getImage();
	imageForHopfield = imgProc.getImageForHopfield()
	print imageForHopfield

	net = HopfieldNetwork(imageForHopfield.shape[0])
	net.trainHebb(imageForHopfield)
	net.initNeurons(np.ones(imageForHopfield.shape[0]))

	net.update(20)
	remeberedImage20 = net.getNeuronsMatrix().copy()
	remeberedImage20 = np.reshape(remeberedImage20, (image.shape[0],image.shape[0]))
	net.update(80)
	remeberedImage100 = net.getNeuronsMatrix().copy()
	remeberedImage100 = np.reshape(remeberedImage100, (image.shape[0],image.shape[0]))
	net.update(150)
	remeberedImage250 = net.getNeuronsMatrix().copy()
	remeberedImage250 = np.reshape(remeberedImage250, (image.shape[0],image.shape[0]))
	net.update(250)
	remeberedImage500 = net.getNeuronsMatrix().copy()
	remeberedImage500 = np.reshape(remeberedImage500, (image.shape[0],image.shape[0]))
	net.update(500)
	remeberedImage1000 = net.getNeuronsMatrix().copy()
	remeberedImage1000 = np.reshape(remeberedImage1000, (image.shape[0],image.shape[0]))

	plt.subplot(241),plt.imshow(image,cmap = 'gray', interpolation = 'nearest')
	plt.title('Orginal image'), plt.xticks([]), plt.yticks([])	

	plt.subplot(242),plt.imshow(remeberedImage20,cmap = 'gray', interpolation = 'nearest')
	plt.title('Image after 20 iterations'), plt.xticks([]), plt.yticks([])
	plt.subplot(243),plt.imshow(remeberedImage100,cmap = 'gray', interpolation = 'nearest')
	plt.title('Image after 100 iterations'), plt.xticks([]), plt.yticks([])
	plt.subplot(244),plt.imshow(remeberedImage250,cmap = 'gray', interpolation = 'nearest')
	plt.title('Image after 250 iterations'), plt.xticks([]), plt.yticks([])
	plt.subplot(245),plt.imshow(remeberedImage500,cmap = 'gray', interpolation = 'nearest')
	plt.title('Image after 500 iterations'), plt.xticks([]), plt.yticks([])
	plt.subplot(246),plt.imshow(remeberedImage1000,cmap = 'gray', interpolation = 'nearest')
	plt.title('Image after 1000 iterations'), plt.xticks([]), plt.yticks([])
	plt.show()

if __name__ == '__main__':
	main()