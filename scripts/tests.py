#!/usr/bin/python
import numpy as np
from os import listdir
from os.path import join
from hopfield import HopfieldNetwork
from imgPreprocessor import ImgPreprocessor
from imageSaver import ImageSaver
from imageDisturber import ImageDisturber

import cv2
import cv

SIDE_OF_ARRAY = 32

NOISE_FACTOR = 0.05

HEBB_PREFIX = 'hebb_'
DELTA_PREFIX = 'delta_'
PINV_PREFIX = 'pinv_'

RESULTS_PATH = 'results/'

def testAllNetworks():
	path = 'signs/'

	images = [path+f for f in listdir(path)]

	print images

	netHebb      = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)
	netPseudoInv = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)
	netDelta     = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)

	for image in images:
		imageName = image
		imgProc = ImgPreprocessor(image,SIDE_OF_ARRAY,127)
		image = imgProc.getImageForHopfield().copy()

		cv2.imshow('test',imgProc.getImage())
		ImageSaver(imageName.replace('signs','results'), imgProc.getImage(),SIDE_OF_ARRAY)

		netHebb.trainHebb(image)
		netDelta.trainDelta(image)
		netPseudoInv.appendTrainingVectorForPseudoInversion(image)

	netPseudoInv.trainPseudoInversion()

	for image in images:
		print "Processing image " + image
		imageName = image
		imgProc = ImgPreprocessor(image,SIDE_OF_ARRAY,127)
		image = imgProc.getImageForHopfield().copy()
		image = ImageDisturber(image, NOISE_FACTOR).get()
		ImageSaver(imageName.replace('signs/','results/rec_pinv_'), image,SIDE_OF_ARRAY)
		netPseudoInv.initNeurons(image)
		netDelta.initNeurons(image)
		netHebb.initNeurons(image)
		
		netHebb.updateUntilSatisfied()
		ImageSaver(imageName.replace('signs/','results/hebb_'), netHebb.getNeuronsMatrix(),SIDE_OF_ARRAY)
		netDelta.updateUntilSatisfied()
		ImageSaver(imageName.replace('signs/','results/delta_'), netDelta.getNeuronsMatrix(),SIDE_OF_ARRAY)
		netPseudoInv.updateUntilSatisfied()
		ImageSaver(imageName.replace('signs/','results/pinv_'), netPseudoInv.getNeuronsMatrix(),SIDE_OF_ARRAY)



if __name__ == '__main__':
	testAllNetworks()
