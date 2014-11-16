#!/usr/bin/python
import numpy as np
import random

class HopfieldNetwork:
	"""A simple implementation of Hopfield Network"""

	def __init__(self, neuronsCount):
		self.neuronsCount = neuronsCount
		self.weightsMatrix = np.array([])
		self.outputMatrix = np.array([])

	def getNeuronsMatrix(self):
		return self.outputMatrix

	def getWeightsMatrix(self):
		return self.weightsMatrix

	def initWeights(self, numberOfNeurons):
		self.weightsMatrix = np.zeros((numberOfNeurons, numberOfNeurons))
		print "Weights of net were initialized."

	def initNeurons(self, desiredNeuronsStateMatrix):
		self.outputMatrix = desiredNeuronsStateMatrix
		print "Network was initilized."

	def update(self, updatesNumber):
		for u in range(updatesNumber):
			neuron = random.randint(0,self.neuronsCount-1)
			#print "Updating neuron: " + str(neuron)
			activation = 0
			for i in range(self.neuronsCount):
				activation += self.outputMatrix[i] * self.weightsMatrix[neuron][i]
				#print "Activation value:" + str(activation)
				self.outputMatrix[neuron] = 1 if activation > 0 else -1
			#print self.outputMatrix

	def trainHebb(self, trainingMatrix, learningRate=0.01):
		print "WeightsMatrix after training"
		for i in range(self.neuronsCount):
			for j in range(self.neuronsCount):
				if i != j:
					self.weightsMatrix[i][j] += learningRate * trainingMatrix[i] * trainingMatrix[j]
		print self.weightsMatrix
