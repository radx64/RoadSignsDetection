import numpy as np
import random

class HopfieldNetwork:
	"""A simple implementation of Hopfield Network"""

	def __init__(self, neuronsCount):
		self.neuronsCount = neuronsCount
		self.weightsMatrix = np.array([])
		self.outputMatrix = np.array([])
		self.initWeights()

	def getNeuronsMatrix(self):
		return self.outputMatrix

	def getWeightsMatrix(self):
		return self.weightsMatrix

	def initWeights(self):
		self.weightsMatrix = np.zeros((self.neuronsCount, self.neuronsCount)).copy()
		print "Weights of net were initialized."

	def initNeurons(self, desiredNeuronsStateMatrix):
		self.outputMatrix = desiredNeuronsStateMatrix.copy()
		print "Network was initialized."

	def update(self, updatesNumber):
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
					print "Neuron " + str(neuron) + " has changed"
			#print self.outputMatrix

	def trainHebb(self, trainingMatrix, learningRate=0.001):
		print "WeightsMatrix after training"
		for i in range(self.neuronsCount):
			for j in range(self.neuronsCount):
				if i != j:
					self.weightsMatrix[i][j] += learningRate * trainingMatrix[i] * trainingMatrix[j]
		print self.weightsMatrix

	def trainPseudoinversion(self):
		print "Not implemented yet! Network state was left unaffected"
		pass
