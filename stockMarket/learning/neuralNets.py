
import os
import pickle as pi
import scipy as sp
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from sklearn import decomposition
from sklearn import lda
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

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
    if i in cvIds:
        cvAnswers.append(allAnswers[i])
        cvFeatures.append(allFeatures[i])
    elif i in testIds:
    # if i+260 in testIds:
        testAnswers.append(allAnswers[i])
        testFeatures.append(allFeatures[i])
    else:
        finalAnswers.append(allAnswers[i])
        finalFeatures.append(allFeatures[i])

#Shuffling Training Features:
shuffled = np.asarray(finalFeatures)
shuffled = np.hstack((shuffled, np.zeros((shuffled.shape[0], 1), dtype=shuffled.dtype)))
shuffled[:, -1] = finalAnswers
np.random.shuffle(shuffled)
finalFeatures = shuffled[:, :-1]
finalAnswers = shuffled[:, -1]

#Applying LDA:
# print 'Applying LDA...'
# n_components = 200
# selLda = lda.LDA(n_components=n_components)
# finalFeatures = selLda.fit_transform(finalFeatures, finalAnswers)
# cvFeatures = selLda.transform(cvFeatures)
# testFeatures = selLda.transform(testFeatures)

#Applying PCA:
# print 'Applying PCA...'
# n_components = 2000
# pca = decomposition.PCA(n_components=n_components)
# finalFeatures = pca.fit_transform(finalFeatures)
# cvFeatures = pca.transform(cvFeatures)
# testFeatures = pca.transform(testFeatures)
# print 'With ' + str(n_components) + ' components we have a conserved variance of ' + str(pca.explained_variance_ratio_)

