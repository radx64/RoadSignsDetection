#!/usr/bin/python

from hopfield import HopfieldNetwork
import numpy as np

trainSet = np.array(
		[
			[-1,-1, 1, 1,-1,-1],
			[ 1, 1,-1,-1, 1, 1]
		])

def main():
	net = HopfieldNetwork(6)
	net.initWeights(6)
	net.trainHebb(trainSet[0])
	net.trainHebb(trainSet[1])
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