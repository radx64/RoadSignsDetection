#!/usr/bin/python

import numpy as np
import random


trainSet = np.array(
		[
			[-1,-1,-1,-1],
			[-1, 1, 1,-1]
		])

print "Prepared trainSet shape: " + str(trainSet.shape)

class HopfieldNetwork:
	"""A simple implementation of Hopfield Network"""

	def __init__(self, neuronsCount, learningRate):
		self.n = neuronsCount
		self.learningRate = learningRate
		self.weightsMatrix = np.array([])
		self.outputMatrix = np.array([])

	def initWeights(self, numberOfNeurons):
		self.weightsMatrix = np.zeros((numberOfNeurons, numberOfNeurons))
		print "Weights of net were initialized."
		print self.weightsMatrix

	def initNetwork(self, desiredNeuronsStateMatrix):
		print self.outputMatrix
		self.outputMatrix = desiredNeuronsStateMatrix
		print self.outputMatrix
		print "Network was initilized with:"
		print desiredNeuronsStateMatrix

	def updateNetwork(self, updatesNumber):
		for u in range(updatesNumber):
			neuron = random.randint(0,self.n-1)
			print "Updating neuron: " + str(neuron)
			activation = 0
			for i in range(self.n):
				activation += self.outputMatrix[i] * self.weightsMatrix[neuron][i]
				print "Activation value:" + str(activation)
				self.outputMatrix[neuron] = 1 if activation > 0 else -1
			print self.outputMatrix

	def train(self, trainingMatrix):
		print "Training net"
		for i in range(self.n):
			for j in range(self.n):
				if i != j:
					self.weightsMatrix[i][j] += self.learningRate * trainingMatrix[i] * trainingMatrix[j]
		print self.weightsMatrix

def main():
	net = HopfieldNetwork(4,0.3)
	net.initWeights(4)
	net.train([-1,-1,1,1])
	net.train([1,1,-1,-1])
	net.initNetwork([1,1,1,-1])
	net.updateNetwork(10)
	pass

if __name__ == '__main__':
	main()