from sklearn.datasets import load_iris
from sklearn import tree 
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load iris data set
iris = load_iris()

x = iris.data
y = iris.target

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# select classifier
my_classifier = tree.DecisionTreeClassifier()
#my_classifier = neighbors.KNeighborsClassifier()



# train
my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)
print(predictions)

# accuracy
print(accuracy_score(y_test, predictions))



