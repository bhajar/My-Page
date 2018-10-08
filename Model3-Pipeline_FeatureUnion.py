import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prediction_data = pd.read_csv('prediction_data_v2.csv', encoding='latin-1')
#prediction_data.iloc[:,9:3025] = prediction_data.iloc[:,9:3025].astype(object)

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

X = pd.DataFrame(prediction_data.iloc[:,1])
X = pd.concat([X, X_transform], axis=1)
y = prediction_data.revenue_class
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

###################
###Feature Union###
###################
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
vect = CountVectorizer()

# Numeric transformation
def get_manual(df):
    return df.iloc[:,1:38]
    #return df.loc[:,['budget','runtime', 'vote_average', 'vote_count', 'polarity', 'afinn']] 
get_manual(X_train).head()
from sklearn.preprocessing import FunctionTransformer
get_manual_ft = FunctionTransformer(get_manual, validate=False)
get_manual_ft.transform(X_train).shape

# Text transformation
def get_text(df):
    return df.cleaned_reviews
get_text_ft = FunctionTransformer(get_text, validate=False)
get_text_ft.transform(X_train).shape

# Combining
from sklearn.pipeline import make_union
union = make_union(make_pipeline(get_text_ft, vect), get_manual_ft)
X_dtm_manual_train = union.fit_transform(X_train)
X_dtm_manual_train.shape

#--------------------------#
#----Logistic Regression---#
#--------------------------#
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 0)
pipe = make_pipeline(union, logreg)
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) #Test Accuracy

from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean() #CV accuracy

cm = metrics.confusion_matrix(y_test, y_pred_class_pipe)
accuracy_1away = (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]+cm[3,2]+cm[2,3]+cm[3,3]+cm[4,3]+cm[3,4]+cm[4,4]+cm[4,5]+cm[5,4]
                  +cm[5,5]+cm[5,6]+cm[6,5]+cm[6,6]+cm[6,7]+cm[7,6]+cm[7,7]+cm[7,8]+cm[8,7]+cm[8,8])/len(y_test)
accuracy_1away 

#-----------------#
#----Kernel SVM---#
#-----------------#
from sklearn.svm import SVC
ksvm = SVC(kernel = 'rbf', random_state = 0)
pipe = make_pipeline(union, ksvm)

from sklearn.model_selection import GridSearchCV
param_grid = {}
param_grid = [{'svc__C': [0.1, 1, 10, 100, 1000], 'svc__kernel': ['rbf']}]
ksvm_grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
ksvm_grid_search = ksvm_grid_search.fit(X_train, y_train)#CV & GridSearch accuracy
best_accuracy = ksvm_grid_search.best_score_
best_parameters = ksvm_grid_search.best_params_

from sklearn.svm import SVC
ksvm = SVC(kernel = 'rbf', C=1) #After tuning
pipe = make_pipeline(union, ksvm)
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) #Test Accuracy

cm = metrics.confusion_matrix(y_test, y_pred_class_pipe)
accuracy_1away = (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]+cm[3,2]+cm[2,3]+cm[3,3]+cm[4,3]+cm[3,4]+cm[4,4]+cm[4,5]+cm[5,4]
                  +cm[5,5]+cm[5,6]+cm[6,5]+cm[6,6]+cm[6,7]+cm[7,6]+cm[7,7]+cm[7,8]+cm[8,7]+cm[8,8])/len(y_test)
accuracy_1away 

#--------------------------#
#----K Nearest Neighbour---#
#--------------------------#
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
pipe = make_pipeline(union, knn)

from sklearn.model_selection import GridSearchCV
param_grid = {}
param_grid = [{'kneighborsclassifier__n_neighbors': [5,10,20,40,80,100]}]
knn_grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
knn_grid_search = knn_grid_search.fit(X_train, y_train)#CV & GridSearch accuracy
best_accuracy = knn_grid_search.best_score_
best_parameters = knn_grid_search.best_params_

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10) #After tuning
pipe = make_pipeline(union, knn)
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) #Test Accuracy

cm = metrics.confusion_matrix(y_test, y_pred_class_pipe)
accuracy_1away = (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]+cm[3,2]+cm[2,3]+cm[3,3]+cm[4,3]+cm[3,4]+cm[4,4]+cm[4,5]+cm[5,4]
                  +cm[5,5]+cm[5,6]+cm[6,5]+cm[6,6]+cm[6,7]+cm[7,6]+cm[7,7]+cm[7,8]+cm[8,7]+cm[8,8])/len(y_test)
accuracy_1away 
