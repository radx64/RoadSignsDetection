import numpy as np
import random
import json

class HopfieldNetwork:
	"""A simple implementation of Hopfield Network"""

	def __init__(self, neuronsCount):
		self.neuronsCount = neuronsCount
		self.weightsMatrix = np.array([])
		self.outputMatrix = np.array([])
		self.learnMatrix = np.array([])	#for pseudoInversion learing method
		self.initWeights()

	def getNeuronsMatrix(self):
		return self.outputMatrix

	def getWeightsMatrix(self):
		return self.weightsMatrix

	def saveNetworkConfigToFile(self, filename):
		config = dict()
		config['neuronsCount'] = self.neuronsCount
		config['weightsMatrix'] = self.getWeightsMatrix().tolist()

		with open(filename, 'w') as outfile:
			json.dump(config, outfile)

	def loadNetworkConfigFromFile(self, filename):
		pass


	def initWeights(self):
		self.weightsMatrix = np.zeros((self.neuronsCount, self.neuronsCount)).copy()
		#print "Weights of net were initialized."

	def initNeurons(self, desiredNeuronsStateMatrix):
		self.outputMatrix = desiredNeuronsStateMatrix.copy()
		#print "Network was initialized."

	def update(self, updatesNumber):
		updatesDone = 0
		for u in range(updatesNumber):
			neuron = random.randint(0,self.neuronsCount-1)
			beforeValue = self.outputMatrix[neuron].copy()
			activation = 0
			for i in range(self.neuronsCount):
				activation += self.outputMatrix[i] * self.weightsMatrix[neuron][i]
				#print "Activation value:" + str(activation)
				self.outputMatrix[neuron] = 1 if activation > 0 else -1
				#print "Before: " + str(beforeValue) + " After: "+ str(self.outputMatrix[neuron]) 
			afterValue = self.outputMatrix[neuron]
			if beforeValue != afterValue:
				updatesDone +=1
				pass
				#print "Neuron " + str(neuron) + " has changed"
			#print self.outputMatrix
		return updatesDone

	def updateUntilSatisfied(self):
		lastRoundUpdates = -1
		while(lastRoundUpdates != 0):
			lastRoundUpdates = self.update(self.neuronsCount*10);
			print "Last updated " + str(lastRoundUpdates) + " neurons"


	def trainHebb(self, trainingVector, learningRate=0.01):
		#print "WeightsMatrix after training"
		for i in range(self.neuronsCount):
			for j in range(self.neuronsCount):
				if i != j:
					self.weightsMatrix[i][j] += learningRate * trainingVector[i] * trainingVector[j]
		#print self.weightsMatrix

	def appendTrainingVectorForPseudoInversion(self, trainingVector):
		if len(self.learnMatrix) == 0:
			self.learnMatrix = trainingVector
		else:
			self.learnMatrix = np.vstack([self.learnMatrix, trainingVector])

	def trainPseudoInversion(self):
		X = self.learnMatrix
		self.weightsMatrix = np.dot(np.linalg.pinv(X),X)

	def trainDelta(self, trainingVector, learningRate=0.8):
		x = np.mat(trainingVector).transpose()
		W = np.mat(self.weightsMatrix)
		result = learningRate/self.neuronsCount * ((x - W * x) * x.transpose())
		self.weightsMatrix = np.asarray(W + result)
