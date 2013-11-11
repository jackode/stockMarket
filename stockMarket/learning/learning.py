#-*- coding: utf-8 -*-

import pickle as pi
import scipy as sp
import numpy as np
from sklearn import svm

#Gets content of p.pickle, performs the learning task and evaluates itself
print 'Openning files...'
file = open('dayFeatures.pi', 'rb')
dayFeatures = np.asarray(pi.load(file))
file = open('cvIds.pi', 'rb')
cvIds = pi.load(file)
file = open('testIds.pi', 'rb')
testIds = pi.load(file)
file = open('cvAnswers.pi', 'rb')
cvAnswers = pi.load(file)
file = open('finalAnswers.pi', 'rb')
finalAnswers = pi.load(file)
# file = open('testAnswers.pi', 'rb')
# testAnswers = pi.load(file)


finalFeatures = []
testFeatures = []
cvFeatures = []
print 'Building features inputs...'
for i in xrange(260, len(dayFeatures)):
    feature = np.array([])
    for j in xrange(0, 260):
        feature = np.append(feature, dayFeatures[i-j])
    if i in cvIds:
        cvFeatures.append(feature)
    elif i in testIds:
        testFeatures.append(feature)
    else:
        finalFeatures.append(feature)

#Training the SVM:
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
print 'The SVM Performed ' + str(100*float(evaluation)/float(len(cvFeatures))) + '% on the test set.'