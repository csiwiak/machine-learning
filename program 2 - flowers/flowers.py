import via as np
from sklearn.datasets import load_iris
from sklearn import tree 


# load iris data set
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])


# split data for testing
# remove 3 enties from data set and use them for tesing
test_idx = [0,50,100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# train
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# predict
print(test_target)
print(clf.predict(test_data))

# visualize tree



