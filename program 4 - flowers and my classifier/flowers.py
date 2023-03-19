import random

from sklearn.datasets import load_iris
from sklearn import tree 
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

class MyRandomClassifier():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions =[]
        for row in X_test:
            label = random.choice(self.Y_train)
            predictions.append(label)
        return predictions






def euc(a, b):
    return distance.euclidean(a, b)

class MyKNNClassifier():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions =[]
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]








# load iris data set
iris = load_iris()

x = iris.data
y = iris.target

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# select classifier
#my_classifier = tree.DecisionTreeClassifier()
#my_classifier = neighbors.KNeighborsClassifier()
#my_classifier = MyRandomClassifier()
#my_classifier = MyClassifier()
my_classifier = MyKNNClassifier()

# train
my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)
print(predictions)

# accuracy
print(accuracy_score(y_test, predictions))


