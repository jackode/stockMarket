#-*- coding: utf-8 -*-

import pickle as pi
import scipy as sp
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition
from sklearn import lda
from sklearn import neighbors
from sklearn.feature_selection import RFE

#Gets content of p.pickle, performs the learning task and evaluates itself
print 'Openning files...'
file = open('dayFeatures.pi', 'rb')
dayFeatures = np.asarray(pi.load(file))
file = open('cvIds.pi', 'rb')
cvIds = pi.load(file)
file = open('testIds.pi', 'rb')
testIds = pi.load(file)
file = open('allAnswers.pi', 'rb')
allAnswers = pi.load(file)
# file = open('cvAnswers.pi', 'rb')
# cvAnswers = pi.load(file)
# file = open('finalAnswers.pi', 'rb')
# finalAnswers = pi.load(file)
# file = open('testAnswers.pi', 'rb')
# testAnswers = pi.load(file)

#Scaling the features:
print 'Scaling features...'
for i in xrange(len(dayFeatures[0])):
    dayFeatures[:, i] = (dayFeatures[:, i] - np.average(dayFeatures[:, i])) / (np.amax(dayFeatures[:, i]) - np.amin(dayFeatures[:, i]))

#Building features inputs
allFeatures = []
print 'Building features inputs...'
for i in xrange(260, len(dayFeatures)):
    feature = np.array([])
    for j in xrange(0, 260):
        feature = np.append(feature, dayFeatures[i-j])
    allFeatures.append(feature)

#Building Sets:
print 'Building sets...'
finalFeatures = []
testFeatures = []
cvFeatures = []
finalAnswers = []
testAnswers = []
cvAnswers = []
for i in xrange(0, len(allFeatures)):
    if i+260 in cvIds:
        cvAnswers.append(allAnswers[i])
        cvFeatures.append(allFeatures[i])
    elif i+260 in testIds:
        testAnswers.append(allAnswers[i])
        testFeatures.append(allFeatures[i])
    else:
        finalAnswers.append(allAnswers[i])
        finalFeatures.append(allFeatures[i])

#Applying LDA:
print 'Applying LDA...'
n_components = 200
selLda = lda.LDA(n_components=n_components)
finalFeatures = selLda.fit_transform(finalFeatures, finalAnswers)
cvFeatures = selLda.transform(cvFeatures)
testFeatures = selLda.transform(testFeatures)

#Applying PCA:
# print 'Applying PCA...'
# n_components = 2000
# pca = decomposition.PCA(n_components=n_components)
# finalFeatures = pca.fit_transform(finalFeatures)
# cvFeatures = pca.transform(cvFeatures)
# testFeatures = pca.transform(testFeatures)
# print 'With ' + str(n_components) + ' components we have a conserved variance of ' + str(pca.explained_variance_ratio_)

#Training Logistic Regression
print 'Training Logistic Regression'
lrLearner = LogisticRegression()
lrLearner.fit(finalFeatures, finalAnswers)

#Evaluating Logistic Regression:
evaluation = 0
print 'Evaluating Logistic Regression...'
for i in xrange(len(cvFeatures)):
    prediction = lrLearner.predict(cvFeatures[i])
    if prediction == cvAnswers[i]:
        evaluation += 1
print 'Logistic Regression performed ' + str(100*float(evaluation)/float(len(cvFeatures))) + '% on the test set.'


# Training the SVM:
print 'Training the SVM...'
svmLearner = svm.SVC()
svmLearner.fit(finalFeatures, finalAnswers)

#Evaluating the SVM:
evaluation = 0
print 'Evaluating the SVM...'
for i in xrange(len(cvFeatures)):
    prediction = svmLearner.predict(cvFeatures[i])
    if prediction == cvAnswers[i]:
        evaluation += 1
print 'The SVM performed ' + str(100*float(evaluation)/float(len(cvFeatures))) + '% on the test set.'

#Training K-Nearest Neigbors
print 'Training k-NN'
knnLearner = neighbors.KNeighborsClassifier(n_neighbors=350)
knnLearner.fit(finalFeatures, finalAnswers)

#Evaluating k-NN:
evaluation = 0
print 'Evaluating k-NN...'
for i in xrange(len(cvFeatures)):
    prediction = knnLearner.predict(cvFeatures[i])
    if prediction == cvAnswers[i]:
        evaluation += 1
print 'k-NN performed ' + str(100*float(evaluation)/float(len(cvFeatures))) + '% on the test set.'


#Applying Wrappers for Logistic Regression:
# print 'Applyig Wrappers...'
# selWrapper = RFE(lrLearner, n_features_to_select=50)
# selWrapper.fit(cvFeatures, cvAnswers)
# print selWrapper.ranking_
