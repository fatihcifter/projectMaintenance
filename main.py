import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

dataset = pd.read_csv('C:/Users/fatihcifter/PycharmProjects/pythonProject/content/test.csv')  # dataset yolu
dataset.info()
print(dataset.head())
print(np.unique(dataset[['StockId']].values))

standardScaler = StandardScaler()
columns_to_scale = ['WorkStation', 'StockId', 'Fire', 'Equipment']

dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
print(dataset.head())
irisData = load_iris()
le = preprocessing.LabelEncoder()
datasetTarget = le.fit_transform(dataset['ErrorID'])
dataset['ErrorID'] = datasetTarget
X = dataset[columns_to_scale]
y = dataset['ErrorID']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
neighbors = np.arange(1, 40)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
knn_scores = []
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn_scores.append(knn.score(X_test, y_test))
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
print(f'en iyi seçim: {np.argmax(knn_scores)+1}')
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')


plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
