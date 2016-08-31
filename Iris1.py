# Iris Classification using K Nearest Neighbour model
#	Made By - Anshuman Dixit

import csv
import random
import math
import operator
 
##################### Loading and splitting Dataset ######################### 

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])

##################### Finding Distances between two instances ###############	            

def euclideanDistance(instance1,instance2,length):
	distance = 0
	for x in range(length):
		distance+=pow((instance1[x] - instance2[x]),2)
	return math.sqrt(distance)

##################### Getting K nearest neighbours ##########################	

def getNeighbours(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance) - 1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances. sort(key=operator.itemgetter(1))
	neighbours = []
	for x in range(k):
		neighbours.append(distances[x][0])
	return neighbours

##################### Getting the Prediction ################################	

def getResponse(neighbours):
	classVotes = {}
	for x in range(len(neighbours)):
		response = neighbours[x][-1]
		if response in classVotes:
			classVotes[response] +=1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse = True)
	return sortedVotes[0][0]

#################### Getting the Accuracy ###################################	

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

#################### Main function ##########################################	    

def main():
	# preparing data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('iris.data.csv', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	#generating predictions
	predictions = []
	k = 3
	for x in range(len(testSet)):
		neighbours = getNeighbours(trainingSet, testSet[x], k)
		result = getResponse(neighbours)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))# 107,43
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%') # 97.2972972972973%
main()
