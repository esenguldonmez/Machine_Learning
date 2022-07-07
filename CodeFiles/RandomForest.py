import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import datasets

# Merged Datas for training and testing
path = "Datas.csv"
# Read datas from csv file
dataset = pd.read_csv(path)
dataset.head()

# get all datas except column name and patient number
X = dataset.iloc[:, 2:60].values
# Define our target column (disease_type)
y = dataset.iloc[:, -1].values

# Split dataset into training set (%100) and test set (%30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# Create RandomFores
classifier = RandomForestClassifier(n_estimators = 50)
# Train the model using the training sets y_pred=clf.predict(X_test)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# print Confusion matrix
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)

# print Classification Report
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)

# Model Accuracy, how often is the classifier correct?
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

sensitivity = result[0,0]/(result[0,0]+result[0,1])
print('Sensitivity : ', sensitivity )

specificity = result[1,1]/(result[1,0]+result[1,1])
print('Specificity : ', specificity)

# Number of samples
BreastNumberOfSamples = result[0, 0] + result[0, 1] + result[0, 2] + result[0, 3]
ColonNumberOfSamples = result[1, 0] + result[1, 1] + result[1, 2] + result[1, 3]
LungNumberOfSamples = result[2, 0] + result[2, 1] + result[2, 2] + result[2, 3]
ProstateNumberOfSamples = result[3, 0] + result[3, 1] + result[3, 2] + result[3, 3]

# Statistic for Breast Cancer
print('-------------------Breast Cancer-------------------')
BreastSensitivity = result[0, 0 ]/ BreastNumberOfSamples
print('BreastSensitivity : ', BreastSensitivity)
BreastSpecificity = (result[1, 1] + result[2, 2] + result[3, 3]) / (ColonNumberOfSamples+LungNumberOfSamples+ProstateNumberOfSamples)
print('BreastSpecificity : ', BreastSpecificity)

# Statistic for Colon Cancer
print('-------------------Colon Cancer-------------------')
ColonSensitivity = result[1,1]/ ColonNumberOfSamples
print('ColonSensitivity : ', ColonSensitivity)
ColonSpecificity = (result[0, 0] + result[2, 2] + result[3, 3]) / (BreastNumberOfSamples+LungNumberOfSamples+ProstateNumberOfSamples)
print('ColonSpecificity : ', ColonSpecificity)

# Statistic for Lung Cancer
print('-------------------Lung Cancer-------------------')
LungSensitivity = result[2,2]/ LungNumberOfSamples
print('LungSensitivity : ', LungSensitivity)
LungSpecificity = (result[0, 0] + result[1,1] + result[3, 3]) / (BreastNumberOfSamples+ColonNumberOfSamples+ProstateNumberOfSamples)
print('LungSpecificity : ', LungSpecificity)

# Statistic for Prostate Cancer
print('-------------------Prostate Cancer-------------------')
ProstateSensitivity = result[3,3]/ ProstateNumberOfSamples
print('ProstateSensitivity : ', ProstateSensitivity)
ProstateSpecificity = (result[0, 0] + result[1,1] + result[2,2]) / (BreastNumberOfSamples+ColonNumberOfSamples+LungNumberOfSamples)
print('ProstateSpecificity : ', ProstateSpecificity)