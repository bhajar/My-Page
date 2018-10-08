import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk

final_data = pd.read_csv('final_data.csv', encoding='latin-1')
cleaned_reviews = pd.read_csv('cleaned_reviews.csv')
final_data['cleaned_reviews'] = cleaned_reviews

# Splitting the dataset into the Training set and Test set
X = final_data.cleaned_reviews
y = final_data.revenue_class
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

##########################################
### Bag of Words & Text Classification ###
##########################################
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
#from sklearn.feature_extraction.text import TfidfVectorizer
#vect = TfidfVectorizer()

X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape
X_test_dtm = vect.transform(X_test)
X_test_dtm.shape

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# define a function that accepts a vectorizer and calculates the accuracy
def tokenize_test(vect):
    # create document-term matrices using the vectorizer
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    # print the number of features that were generated
    print('Features: ', X_train_dtm.shape[1])
    # use Multinomial Naive Bayes to predict the star rating
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    # print the accuracy of its predictions
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))
print(vect.get_feature_names())

vect = CountVectorizer()
#vect = TfidfVectorizer(min_df=2)
tokenize_test(vect)

# Confusion Matrix NB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)
metrics.confusion_matrix(y_test, y_pred_class)

#############################################
### Review Analysis - Model with Pipeline ###
#############################################
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
vect = CountVectorizer()

#----------------------------#
#---Identifying best model---#
#----------------------------#
#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
pipe = make_pipeline(vect, nb)
pipe.steps
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) # 0.30126002290950743


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 0)
pipe = make_pipeline(vect, logreg)
pipe.steps
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) # 0.31729667812142037

#kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
pipe = make_pipeline(vect, knn)
pipe.steps
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) # 0.26575028636884307

#Linear SVM
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', random_state = 0)
pipe = make_pipeline(vect, svm)
pipe.steps
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) # 0.31958762886597936

#Kernel SVM
from sklearn.svm import SVC
ksvm = SVC(kernel = 'rbf', random_state = 0, C = 1)
pipe = make_pipeline(vect, ksvm)
pipe.steps
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) # 0.34020618556701032

#RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
pipe = make_pipeline(vect, rf)
pipe.steps
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) #0.3138602520045819

#One-Away accuracy
cm = metrics.confusion_matrix(y_test, y_pred_class_pipe)
accuracy_1away = (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]+cm[3,2]+cm[2,3]+cm[3,3]+cm[4,3]+cm[3,4]+cm[4,4]+cm[4,5]+cm[5,4]
                  +cm[5,5]+cm[5,6]+cm[6,5]+cm[6,6]+cm[6,7]+cm[7,6]+cm[7,7]+cm[7,8]+cm[8,7]+cm[8,8])/len(y_test)
accuracy_1away 
#0.57502863688430694 for Kernel SVM

#---------------------------------#
#---Grid Search with Best Model---#
#---------------------------------#
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
vect = CountVectorizer()

#Kernel SVM
from sklearn.svm import SVC
ksvm = SVC(kernel = 'rbf', random_state = 0)
pipe = make_pipeline(vect, ksvm)
pipe.steps
pipe.fit(X_train, y_train)
y_pred_class_pipe = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class_pipe) 

from sklearn.model_selection import GridSearchCV
parameters = [{'svc__C': [0.1, 1, 10, 50, 100, 500 ,1000], 'svc__kernel': ['rbf']}]

grid_search = GridSearchCV(estimator = pipe,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

############################
###Finalizing the dataset###
############################
final_data['cleaned_reviews'] = pd.read_csv('cleaned_reviews.csv', encoding='latin-1')
prediction_data = final_data[['revenue_class','cleaned_reviews','budget','runtime', 'vote_average', 'vote_count', 'polarity', 'afinn',
                              'genres_1_name', 'production_companies_1_name', 'production_country_name', 
                              'director_name']]
#prediction_data.to_csv('prediction_data_v1.csv',index=False)
#See R for OneHotEncoder & v2