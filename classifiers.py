import numpy as np
from sklearn.naive_bayes import GaussianNB
import sklearn.svm as svm
from sklearn.tree import DecisionTreeClassifier

"""
Use Pandas for reading data into a Pandas DataFrame
Use native Pandas features to process text features to numerical ones
Use the extremely convenient "dummies" feature of the Pandas library to convert categorical features to binary ones. 
(One-Hot encoding).
 Scikit has it's own One-Hot Encoding routine but it only works with integers 
 (Features with categories like 1,2,3 rather than 'a','b','c'). Pandas can digest anything thrown at it.
Finally, explicitly cast the DataFrame into a numpy array which can be used  by the scikit-learn API.
 Note that at this point you lose your feature labels (Headers), 
 so it would be difficult to keep track of the features if you use the "feature-importance" routine in scikit-learn. 
 I have the practice of saving the headers before casting the data-frame into a numpy array. 
 [>>list(<DataFrame>) prints out the headers into a nice list]
"""

def naive_bayes(x_vectors, x_labels, y_vectors):
    clf = GaussianNB()
    clf.fit(x_vectors, x_labels)
    predictions = (clf.predict(y_vectors))
    probabilities = clf.predict_proba(y_vectors) 
    return predictions, probabilities


def decision_tree_classifier(x_vectors, x_labels, y_vectors):
    clf = DecisionTreeClassifier()
    clf.fit(x_vectors, x_labels)
    predictions = (clf.predict(y_vectors))
    probabilities = clf.predict_proba(y_vectors) 
    return predictions, probabilities

def svm_classifier(x_vectors, x_labels, y_vectors):
    clf = svm.LinearSVC()
    clf.fit(x_vectors, x_labels)
    predictions = (clf.predict(y_vectors))
    probabilities = clf.decision_function(y_vectors) 
    return predictions, probabilities



# Testing below

#TO DO
#write to json format such thast we get (tweet#: [label, confidence])
#output to output/classifier_output