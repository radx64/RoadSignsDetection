#!/usr/bin/python
import numpy as np
import time
from os import listdir
from os.path import join
from hopfield import HopfieldNetwork
from imgPreprocessor import ImgPreprocessor
from imageSaver import ImageSaver
from imageNoise import ImageNoise
from imageLines import ImageLines

import cv2
import cv

SIDE_OF_ARRAY = 32

HEBB_PREFIX = 'hebb_'
DELTA_PREFIX = 'delta_'
PINV_PREFIX = 'pinv_'

RESULTS_PATH = 'results/'

RESULT_FILE = 'results.html'

def appendHtmlHeader(file):
	file.write("<!DOCTYPE html>\n")
	file.write("<html>\n")
	file.write("<head>\n")
	file.write("<link rel=\"stylesheet\" href=\"cleanTable.css\">\n")
	file.write("</head>\n")
	file.write("<body>\n")
	file.write("<table>\n")
	file.write("<tr>\n")
	file.write("<th>Znak</th>\n")
	file.write("<th>Parametr zaburzenia</th>\n")
	file.write("<th>Zaburzony znak</th>\n")
	file.write("<th>Hebb</th>\n")
	file.write("<th>Delta</th>\n")
	file.write("<th>Pinv</th>\n")
	file.write("</tr>\n")

def appendHtmlFooter(file):
	file.write("</tables>\n")
	file.write("</body>\n")
	file.write("</html>\n")

def testAllNetworks():
	fileHandle = open(RESULT_FILE, 'w', 0)
	appendHtmlHeader(fileHandle)

	path = 'signs/'

	images = [path+f for f in listdir(path)]

	print images

	netHebb      = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)
	netPseudoInv = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)
	netDelta     = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)

	for image in images:
		print "Learning image " + image
		imageName = image
		imgProc = ImgPreprocessor(image,SIDE_OF_ARRAY,127)
		image = imgProc.getImageForHopfield().copy()

		cv2.imshow('test',imgProc.getImage())
		ImageSaver(imageName.replace('signs','results'), imgProc.getImage(),SIDE_OF_ARRAY)
		netHebb.trainHebb(image)
		netDelta.trainDelta(image)
		netPseudoInv.appendTrainingVectorForPseudoInversion(image)

	netPseudoInv.trainPseudoInversion()

	print "Learning done"
	#noise processing
	for image in images:
		tmpImage = image
		for noiseFactor in [0.05, 0.10, 0.15]:
			image = tmpImage
			fileHandle.write('<tr>\n')
			start = time.time()
			print "Noise factor: " + str(int(noiseFactor*100)) + "%"
			print "Processing image " + image

			imageName = image

			fileHandle.write('<td>\n')
			fileHandle.write(imageName.replace('signs/','').replace('.png','') +'     '+ '<img src=' + imageName.replace('signs','results') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')
			
			fileHandle.write('<td>\n')
			fileHandle.write('Szum: ' + str(int(noiseFactor*100)) +'%')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')

			imgProc = ImgPreprocessor(image,SIDE_OF_ARRAY,127)
			image = imgProc.getImageForHopfield().copy()
			image = ImageNoise(image, noiseFactor).get()
			ImageSaver(imageName.replace('signs/','results/rec_noise_' + str(int(noiseFactor*100)) + '_'), image,SIDE_OF_ARRAY)
			
			fileHandle.write('<td>\n')
			fileHandle.write('<img src=' + imageName.replace('signs/','results/rec_noise_' + str(int(noiseFactor*100)) + '_') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')	

			netPseudoInv.initNeurons(image)
			netDelta.initNeurons(image)
			netHebb.initNeurons(image)
			
			netHebb.updateUntilSatisfied()
			ImageSaver(imageName.replace('signs/','results/hebb_noise_' + str(int(noiseFactor*100)) + '_'), netHebb.getNeuronsMatrix(),SIDE_OF_ARRAY)
			netDelta.updateUntilSatisfied()
			ImageSaver(imageName.replace('signs/','results/delta_noise_' + str(int(noiseFactor*100)) + '_'), netDelta.getNeuronsMatrix(),SIDE_OF_ARRAY)
			netPseudoInv.updateUntilSatisfied()
			ImageSaver(imageName.replace('signs/','results/pinv_noise_' + str(int(noiseFactor*100)) + '_'), netPseudoInv.getNeuronsMatrix(),SIDE_OF_ARRAY)
			end = time.time()

			fileHandle.write('<td>\n')
			fileHandle.write('<img src=' + imageName.replace('signs/','results/hebb_noise_' + str(int(noiseFactor*100)) + '_') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')

			fileHandle.write('<td>\n')
			fileHandle.write('<img src=' + imageName.replace('signs/','results/delta_noise_' + str(int(noiseFactor*100)) + '_') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')	

			fileHandle.write('<td>\n')
			fileHandle.write('<img src=' + imageName.replace('signs/','results/pinv_noise_' + str(int(noiseFactor*100)) + '_') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')

			print "Time elapsed " + str(int(end-start)) + " [s]"
			fileHandle.write('</tr>\n')

	#lines processing 
	for image in images:
		tmpImage = image
		for linesEvery in [16,8]:
			image = tmpImage
			fileHandle.write('<tr>\n')
			start = time.time()
			print "Lines every: " + str(int(linesEvery)) 
			print "Processing image " + image
			imageName = image

			fileHandle.write('<td>\n')
			fileHandle.write(imageName.replace('signs/','').replace('.png','') +'     '+ '<img src=' + imageName.replace('signs','results') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')
			
			fileHandle.write('<td>\n')
			fileHandle.write('Linie co: ' + str(int(linesEvery)))
			fileHandle.write('\n')
			fileHandle.write('</td>\n')

			imgProc = ImgPreprocessor(image,SIDE_OF_ARRAY,127)
			image = imgProc.getImageForHopfield().copy()
			image = ImageLines(image, linesEvery).get()
			ImageSaver(imageName.replace('signs/','results/rec_lines_' + str(linesEvery) + '_'), image,SIDE_OF_ARRAY)

			fileHandle.write('<td>\n')
			fileHandle.write('<img src=' + imageName.replace('signs/','results/rec_lines_' + str(linesEvery) + '_') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')	

			netPseudoInv.initNeurons(image)
			netDelta.initNeurons(image)
			netHebb.initNeurons(image)
			
			netHebb.updateUntilSatisfied()
			ImageSaver(imageName.replace('signs/','results/hebb_lines_' + str(linesEvery) + '_'), netHebb.getNeuronsMatrix(),SIDE_OF_ARRAY)
			netDelta.updateUntilSatisfied()
			ImageSaver(imageName.replace('signs/','results/delta_lines_' + str(linesEvery) + '_'), netDelta.getNeuronsMatrix(),SIDE_OF_ARRAY)
			netPseudoInv.updateUntilSatisfied()
			ImageSaver(imageName.replace('signs/','results/pinv_lines_' + str(linesEvery) + '_'), netPseudoInv.getNeuronsMatrix(),SIDE_OF_ARRAY)
			end = time.time()

			fileHandle.write('<td>\n')
			fileHandle.write('<img src=' + imageName.replace('signs/','results/hebb_lines_' + str(linesEvery) + '_') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')

			fileHandle.write('<td>\n')
			fileHandle.write('<img src=' + imageName.replace('signs/','results/delta_lines_' + str(linesEvery) + '_') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')	

			fileHandle.write('<td>\n')
			fileHandle.write('<img src=' + imageName.replace('signs/','results/pinv_lines_' + str(linesEvery) + '_') + '>')
			fileHandle.write('\n')
			fileHandle.write('</td>\n')

			print "Time elapsed " + str(int(end-start)) + " [s]"
			fileHandle.write('</tr>\n')

	appendHtmlFooter(fileHandle)
	fileHandle.close()

if __name__ == '__main__':
	testAllNetworks()
