import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prediction_data = pd.read_csv('prediction_data_v2.csv', encoding='latin-1')

#Variance Threshold Feature Selection
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
X = prediction_data.iloc[:,2:3025]
X_transform = sel.fit_transform(X.iloc[:,:])
features = sel.get_support(indices = True)
features = [column for column in X.iloc[:,features]]
X_transform = pd.DataFrame(X_transform)
X_transform.columns = features
list(X_transform)

X = X_transform
y = prediction_data.revenue_class
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

######################################
###Prediction with numeric features###
######################################
from sklearn import metrics

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
accuracy_1away = (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]+cm[3,2]+cm[2,3]+cm[3,3]+cm[4,3]+cm[3,4]+cm[4,4]+cm[4,5]+cm[5,4]
                  +cm[5,5]+cm[5,6]+cm[6,5]+cm[6,6]+cm[6,7]+cm[7,6]+cm[7,7]+cm[7,8]+cm[8,7]+cm[8,8])/len(y_test)
accuracy_1away 

#k Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
accuracy_1away = (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]+cm[3,2]+cm[2,3]+cm[3,3]+cm[4,3]+cm[3,4]+cm[4,4]+cm[4,5]+cm[5,4]
                  +cm[5,5]+cm[5,6]+cm[6,5]+cm[6,6]+cm[6,7]+cm[7,6]+cm[7,7]+cm[7,8]+cm[8,7]+cm[8,8])/len(y_test)
accuracy_1away 

#SVM Radial
from sklearn.svm import SVC
ksvm = SVC(kernel = 'linear') #After tuning
ksvm.fit(X_train, y_train)
y_pred = ksvm.predict(X_test)
metrics.accuracy_score(y_test, y_pred) #Test Accuracy
cm = metrics.confusion_matrix(y_test, y_pred)
accuracy_1away = (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]+cm[3,2]+cm[2,3]+cm[3,3]+cm[4,3]+cm[3,4]+cm[4,4]+cm[4,5]+cm[5,4]
                  +cm[5,5]+cm[5,6]+cm[6,5]+cm[6,6]+cm[6,7]+cm[7,6]+cm[7,7]+cm[7,8]+cm[8,7]+cm[8,8])/len(y_test)
accuracy_1away 


