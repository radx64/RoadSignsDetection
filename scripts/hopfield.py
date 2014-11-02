#!/usr/bin/python
import numpy as np
import random

trainSet = np.array(
		[
			[-1,-1, 1, 1,-1,-1],
			[ 1, 1,-1,-1, 1, 1]
		])

print "Prepared trainSet shape: " + str(trainSet.shape)

class HopfieldNetwork:
	"""A simple implementation of Hopfield Network"""

	def __init__(self, neuronsCount, learningRate):
		self.neuronsCount = neuronsCount
		self.learningRate = learningRate
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

	def train(self, trainingMatrix):
		print "Training net"
		for i in range(self.neuronsCount):
			for j in range(self.neuronsCount):
				if i != j:
					self.weightsMatrix[i][j] += self.learningRate * trainingMatrix[i] * trainingMatrix[j]
		print self.weightsMatrix

def main():
	net = HopfieldNetwork(4,0.3)
	net.initWeights(6)
	net.train(trainSet[0])
	net.train(trainSet[1])
	net.initNeurons([1, 1, 1, 1,-1,-1])
	net.update(10)
	print "Results after 10 updates:"
	print net.getNeuronsMatrix()
	net.initNeurons([1, 1, 1,-1, 1, 1])
	net.update(10)
	print "Results after 10 updates:"
	print net.getNeuronsMatrix()

if __name__ == '__main__':
	main()
