#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import sys
import npyscreen

from hopfield import HopfieldNetwork
from imgPreprocessor import ImgPreprocessor

class HopfieldApp(npyscreen.NPSApp):
	def loadParameters(self, parametersArray):
		if(len(parametersArray) < 3):
			#print "Need to give more parameters"
			sys.exit(1)
		signs = list()
		imageToRecognize = ""
		#print "Number of signs to learn: " + str(len(parametersArray)-1)
		for idx, sign in enumerate(parametersArray):
			if (idx > 0) and (idx<len(parametersArray)-1):
				signs.append(sign)
				#print "Image to learn: " + str(sign)
		#print "Image to recognize: " + parametersArray[-1]
		imageToRecognize = parametersArray[-1]
		return signs, imageToRecognize

	def invertMatrix(self, matrix):
		invMatrix = matrix.copy()
		invMatrix[invMatrix == -1] = 0
		invMatrix[invMatrix ==  1] = -1
		invMatrix[invMatrix ==  0] = 1
		return invMatrix

	def main(self):
		SIDE_OF_ARRAY = 32

		F  = npyscreen.Form(name = "Road signs recognition via Hopfield Network",)
		F.add(npyscreen.TitleFixedText, name = "Using dimensions: " + str(SIDE_OF_ARRAY) + "x" + str(SIDE_OF_ARRAY),)

		netHebb      = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)
		netPseudoInv = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)
		netDelta     = HopfieldNetwork(SIDE_OF_ARRAY*SIDE_OF_ARRAY)

		signsList, imgToRecogn = self.loadParameters(sys.argv)
		F.add(npyscreen.TitleFixedText, name = "Image to recognize: " + imgToRecogn,)
		images = list()
		for image in signsList:
			imageName = image
			imgProc = ImgPreprocessor(image,SIDE_OF_ARRAY,127)
			image = imgProc.getImageForHopfield().copy()
			#print image
			F.add(npyscreen.TitleFixedText, name = "Image to learn: " + imageName,)
			images.append(image)
			netHebb.trainHebb(image)
			netDelta.trainDelta(image)
			netPseudoInv.appendTrainingVectorForPseudoInversion(image)

		netPseudoInv.trainPseudoInversion()
		F.display()
		imgProc  = ImgPreprocessor(imgToRecogn,SIDE_OF_ARRAY,127)
		imageRec = imgProc.getImageForHopfield()

		netHebb.initNeurons(imageRec)
		netDelta.initNeurons(imageRec)
		netPseudoInv.initNeurons(imageRec)

		hebbProgressWidget = F.add(npyscreen.TitleSlider, out_of=100, name  = "Hebb Updates     ")
		deltaProgressWidget = F.add(npyscreen.TitleSlider, out_of=100, name = "Delta Updates    ")
		pInvProgressWidget = F.add(npyscreen.TitleSlider, out_of=100, name  = "PseudoInv Updates")

		for i in range(50):
			netHebb.update(100)
			hebbProgressWidget.value = 2 + i*2
			F.display()

		for i in range(50):
			netDelta.update(100)
			deltaProgressWidget.value = 2 + i*2
			F.display()

		for i in range(50):
			netPseudoInv.update(100)
			pInvProgressWidget.value = 2 + i*2
			F.display()

		networkResultHebb      = netHebb.getNeuronsMatrix().copy()
		networkResultDelta     = netDelta.getNeuronsMatrix().copy()
		networkResultPseudoInv = netPseudoInv.getNeuronsMatrix().copy()

		netHebb.saveNetworkConfigToFile('networkConfig.txt');
		"""
		#print "================================"
		matchFound = False
		for idx, image in enumerate(images):
			if np.array_equiv(networkResultHebb, image):
				#print "Found matching image as sample " + str(idx)
				matchFound = True
				break
			elif np.array_equiv(self.invertMatrix(networkResultHebb), image):
				#print "Found inverse matching image as sample" + str(idx)
				matchFound = True
				break
		if(not matchFound):
			pass
			#print "Unfortunately no match!"
		#print "================================"
		"""
		subplotIndex = 201
		subplotIndex += 10 * (len(images)+1)
		
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
		image2dResult = np.reshape(networkResultHebb , (SIDE_OF_ARRAY,SIDE_OF_ARRAY))
		plt.imshow(image2dResult, cmap='gray', interpolation = 'nearest')
		plt.xticks([]), plt.yticks([]), plt.title("Returned by net (HEBB)")

		plt.subplot(newLineSubplotIndex+2)
		image2dResult = np.reshape(networkResultDelta, (SIDE_OF_ARRAY,SIDE_OF_ARRAY))
		plt.imshow(image2dResult, cmap='gray', interpolation = 'nearest')
		plt.xticks([]), plt.yticks([]), plt.title("Returned by net (DELTA)")

		plt.subplot(newLineSubplotIndex+3)
		image2dResult = np.reshape(networkResultPseudoInv, (SIDE_OF_ARRAY,SIDE_OF_ARRAY))
		plt.imshow(image2dResult, cmap='gray', interpolation = 'nearest')
		plt.xticks([]), plt.yticks([]), plt.title("Returned by net (PINVERSE)")
		plt.show()	

if __name__ == '__main__':
    App = HopfieldApp()
    App.run()