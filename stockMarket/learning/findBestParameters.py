#-*- coding: utf-8 -*-

import os
import pickle as pi
import scipy as sp
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition
from sklearn import lda
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report

import thread

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
    if i+260 in cvIds:
        cvAnswers.append(allAnswers[i])
        cvFeatures.append(allFeatures[i])
    elif i+260 in testIds:
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
print 'Applying LDA...'
n_components = 200
selLda = lda.LDA(n_components=n_components)
finalFeatures = selLda.fit_transform(finalFeatures, finalAnswers)
cvFeatures = selLda.transform(cvFeatures)
testFeatures = selLda.transform(testFeatures)

###################
# Finding the best parameters:
###################

def findLR():
    lrBestParam = [0]
    #Finding for LR:
    print 'Finding best parameters for Linear Regression...'
    penalty = ['l2'] #['l1', 'l2']
    dual = [False, True] #[False]
    regul = [pow(10, x) for x in xrange(-6, 6)]
    for pen in penalty:
        for d in dual:
            for reg in regul:
                lrLearner = LogisticRegression(penalty=pen, dual=d, C=reg)
                lrLearner.fit(finalFeatures, finalAnswers)
                score = lrLearner.score(cvFeatures, cvAnswers)
                if lrBestParam[0] < score:
                    lrBestParam = [score, pen, d, reg]
    print 'The best parameters are: ' + str(lrBestParam)

def findSVM():
    svmBestParam = [0]
    #Finding for SVM:
    print 'Finding for SVM...'
    kernels = ['rbf'] #['linear', 'rbf', 'poly', 'sigmoid']
    regul = [pow(5, x) for x in xrange(2, 6)]
    degree = xrange(1, 15)
    for k in kernels:
        for reg in regul:
            if k == 'poly':
                for deg in degree:
                    svmLearner = svm.SVC(C=reg, kernel=k, degree=deg)
                    svmLearner.fit(finalFeatures, finalAnswers)
                    score = svmLearner.score(cvFeatures, cvAnswers)
                    if svmBestParam[0] < score:
                        svmBestParam = [score, reg, k, deg]
            else:
                svmLearner = svm.SVC(C=reg, kernel=k)
                svmLearner.fit(finalFeatures, finalAnswers)
                score = svmLearner.score(cvFeatures, cvAnswers)
                if svmBestParam[0] < score:
                    svmBestParam = [score, reg, k]
    print  'The best parameters are: ' + str(svmBestParam)

def findkNN():
    knnBestParam = [0]
    #Finding for kNN:
    print 'Finding for kNN...'
    nb_neighboors = xrange(50 , 500, 50)
    algos = ['ball_tree', 'kd_tree', 'brute']
    for alg in algos:
        for nb in nb_neighboors:
            knnLearner = neighbors.KNeighborsClassifier(n_neighbors=nb, algorithm=alg)
            knnLearner.fit(finalFeatures, finalAnswers)
            score = knnLearner.score(cvFeatures, cvAnswers)
            if knnBestParam[0] < score:
                knnBestParam = [score, nb, alg]
    print 'The best parameters for kNN are: ' + str(knnBestParam)


# thread.start_new_thread(findLR, ())
# thread.start_new_thread(findkNN, ())
# thread.start_new_thread(findSVM, ())
findSVM()