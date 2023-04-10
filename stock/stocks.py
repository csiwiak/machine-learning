from sklearn.datasets import load_iris
from sklearn import tree 
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import numpy as np
from sklearn.utils import Bunch

def openfile(file):
    with open(file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        return list(csvreader)

def extractClosingPrice(data):
    data.pop(0)  # remove first row with label
    return [i[4] for i in data]  # get only closing price from each row


# ['2023-03-30', '1715.56', '1764.19', '1715.56', '1763.29', '26179385'],
# get  '1763.29'

#['1933.08', '1904.51', '1913.25', '1889.84', '1892.32', '1923.15', '1918.6', '1886.24', '1911.61', '1911.5', '1889.82', '1902.25',
def getDataAndTarget(inputList):
    data = []
    target = []

    numberOfFeaturesInRow = 3
    targedInDaysInFuture = 1
    end = len(inputList)

    for index, item in enumerate(inputList):
        if(index + numberOfFeaturesInRow + targedInDaysInFuture < end):
            row = inputList[index:index+numberOfFeaturesInRow+targedInDaysInFuture-1]
            trg = inputList[index+numberOfFeaturesInRow]
            print(row)
            print(trg)
            data.append(row)
            target.append(trg)
    return Bunch(
        data=data,
        target=target,
    )



#open csv and get data
data = openfile('wig20_d.csv')
closePrices = extractClosingPrice(data)
print(closePrices)


stocks = getDataAndTarget(closePrices)
print("stocks data")
print(stocks.data)
print("stocks target")
print(stocks.target)

# load iris data set
#iris = load_iris()
#x = iris.data
#y = iris.target

x = stocks.data
y = stocks.target

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




