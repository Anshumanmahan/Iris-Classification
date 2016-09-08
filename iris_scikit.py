from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import random
iris = load_iris()
trainx, trainy, testx, testy = [],[],[],[]
for i in range(0,150):
	if random.random() < 0.50:
		trainx.append(iris.data[i])
		trainy.append(iris.target[i])
	else:
		testx.append(iris.data[i])
		testy.append(iris.target[i])
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(trainx , trainy)
ans = knn.predict(testx)
correct = 0
for i in range(len(testy)):
	if ans[i] == testy[i]:
		correct += 1

print (correct/float(len(testy))) * 100.0
