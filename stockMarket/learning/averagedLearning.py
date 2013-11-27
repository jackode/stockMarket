#-*- coding: utf-8 -*-

import os
import pickle as pi
import scipy as sp
import numpy as np


curr_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
execfile(curr_dir + '/../Scripts/plot.py')
execfile(curr_dir + '/learning/learning.py')

def buildSet():
    execfile(curr_dir + '/learning/buildSets.py')

def learnFromSet():
    #Build and Work on features
    dayFeatures, cvIds, testIds, allAnswers = openFiles()
    dayFeatures = scaleFeatures(dayFeatures)
    allFeatures = buildFeaturesInputs(dayFeatures)
    finalFeatures, finalAnswers, cvFeatures, cvAnswers, testFeatures, testAnswers = buildSets(allFeatures, allAnswers)
    finalFeatures, finalAnswers = shuffleSet(finalFeatures, finalAnswers)
    # finalFeatures, cvFeatures, testFeatures = applyLDA(finalFeatures, finalAnswers, cvFeatures, testFeatures, 2000)
    finalFeatures, cvFeatures, testFeatures = applyPCA(finalFeatures, cvFeatures, testFeatures, n_components=2000)

    #Convert features and answers into arrays:
    testFeatures = np.asarray(testFeatures)
    testAnswers = np.asarray(testAnswers)
    finalFeatures = np.asarray(finalFeatures)
    finalAnswers = np.asarray(finalAnswers)
    cvFeatures = np.asarray(cvFeatures)
    cvAnswers = np.asarray(cvAnswers)

    #Creation of the learners:
    lrLearner = LogisticRegression(penalty='l2', dual=False, C=10000.0)
    svmLearner = svm.SVC(C=5, kernel='poly', degree=4, probability=True)
    knnLearner = neighbors.KNeighborsClassifier(n_neighbors=250, algorithm='auto')

    return findTrainerErrorParallel(lrLearner, svmLearner, knnLearner)



# file = open('lrPrecDown', 'rb')
# avelrPrecDown = np.asarray(pi.load(file))

# file = open('lrRecallDown', 'rb')
# avelrRecallDown = np.asarray(pi.load(file))

# file = open('svmPrecDown', 'rb')
# avesvmPrecDown = np.asarray(pi.load(file))

# file = open('svmRecallDown', 'rb')
# avesvmRecallDown = np.asarray(pi.load(file))

# file = open('knnPrecDown', 'rb')
# aveknnPrecDown = np.asarray(pi.load(file))

# file = open('knnRecallDown', 'rb')
# aveknnRecallDown = np.asarray(pi.load(file))

# file = open('lrRecallUp', 'rb')
# avelrRecallUp = np.asarray(pi.load(file))

# file = open('lrPrecUp', 'rb')
# avelrPrecUp = np.asarray(pi.load(file))

# file = open('svmRecallUp', 'rb')
# avesvmRecallUp = np.asarray(pi.load(file))

# file = open('svmPrecUp', 'rb')
# avesvmPrecUp = np.asarray(pi.load(file))

# file = open('knnRecallUp', 'rb')
# aveknnRecallUp = np.asarray(pi.load(file))

# file = open('knnPrecUp', 'rb')
# aveknnPrecUp = np.asarray(pi.load(file))

file = open('lrIndices', 'rb')
avelrIndices = np.asarray(pi.load(file))

file = open('lrTrainingError', 'rb')
avelrTrainingError = np.asarray(pi.load(file))

file = open('lrTestingError', 'rb')
avelrTestingError = np.asarray(pi.load(file))

file = open('svmIndices', 'rb')
avesvmIndices = np.asarray(pi.load(file))

file = open('svmTrainingError', 'rb')
avesvmTrainingError = np.asarray(pi.load(file))

file = open('svmTestingError', 'rb')
avesvmTestingError = np.asarray(pi.load(file))

file = open('knnIndices', 'rb')
aveknnIndices = np.asarray(pi.load(file))

file = open('knnTrainingError', 'rb')
aveknnTrainingError = np.asarray(pi.load(file))

file = open('knnTestingError', 'rb')
aveknnTestingError = np.asarray(pi.load(file))

nb_runs = 20

for i in xrange(1, nb_runs):
    buildSet()
    lrLearner, svmLearner, knnLearner, lrTrainingError, lrTestingError, lrIndices, svmTrainingError, svmTestingError, svmIndices, knnTrainingError, knnTestingError, knnIndices = learnFromSet()

    # file = open('lrPrecDown', 'rb')
    # lrPrecDown = np.asarray(pi.load(file))
    # avelrPrecDown += lrPrecDown

    # file = open('lrRecallDown', 'rb')
    # lrRecallDown = np.asarray(pi.load(file))
    # avelrRecallDown += lrRecallDown

    # file = open('svmPrecDown', 'rb')
    # svmPrecDown = np.asarray(pi.load(file))
    # avesvmPrecDown += svmPrecDown

    # file = open('svmRecallDown', 'rb')
    # svmRecallDown = np.asarray(pi.load(file))
    # avesvmRecallDown += svmRecallDown

    # file = open('knnPrecDown', 'rb')
    # knnPrecDown = np.asarray(pi.load(file))
    # aveknnPrecDown += knnPrecDown

    # file = open('knnRecallDown', 'rb')
    # knnRecallDown = np.asarray(pi.load(file))
    # aveknnRecallDown += knnRecallDown

    # file = open('lrRecallUp', 'rb')
    # lrRecallUp = np.asarray(pi.load(file))
    # avelrRecallUp += lrRecallUp

    # file = open('lrPrecUp', 'rb')
    # lrPrecUp = np.asarray(pi.load(file))
    # avelrPrecUp += lrPrecUp

    # file = open('svmRecallUp', 'rb')
    # svmRecallUp = np.asarray(pi.load(file))
    # avesvmRecallUp += svmRecallUp

    # file = open('svmPrecUp', 'rb')
    # svmPrecUp = np.asarray(pi.load(file))
    # avesvmPrecUp += svmPrecUp

    # file = open('knnRecallUp', 'rb')
    # knnRecallUp = np.asarray(pi.load(file))
    # aveknnRecallUp += knnRecallUp

    # file = open('knnPrecUp', 'rb')
    # knnPrecUp = np.asarray(pi.load(file))
    # aveknnPrecUp += knnPrecUp

    # file = open('lrIndices', 'rb')
    # lrIndices = np.asarray(pi.load(file))
    avelrIndices += lrIndices

    # file = open('lrTrainingError', 'rb')
    # lrTrainingError = np.asarray(pi.load(file))
    avelrTrainingError += lrTrainingError

    # file = open('lrTestingError', 'rb')
    # lrTestingError = np.asarray(pi.load(file))
    avelrTestingError += lrTestingError

    # file = open('svmIndices', 'rb')
    # svmIndices = np.asarray(pi.load(file))
    avesvmIndices += svmIndices

    # file = open('svmTrainingError', 'rb')
    # svmTrainingError = np.asarray(pi.load(file))
    avesvmTrainingError += svmTrainingError

    # file = open('svmTestingError', 'rb')
    # svmTestingError = np.asarray(pi.load(file))
    avesvmTestingError += svmTestingError

    # file = open('knnIndices', 'rb')
    # knnIndices = np.asarray(pi.load(file))
    aveknnIndices += knnIndices

    # file = open('knnTrainingError', 'rb')
    # knnTrainingError = np.asarray(pi.load(file))
    aveknnTrainingError += knnTrainingError

    # file = open('knnTestingError', 'rb')
    # knnTestingError = np.asarray(pi.load(file))
    aveknnTestingError += knnTestingError


#Averaging Everything:
# avelrPrecDown = avelrPrecDown / nb_runs
# avelrRecallDown = avelrRecallDown / nb_runs
# avesvmPrecDown = avesvmPrecDown / nb_runs
# avesvmRecallDown = avesvmRecallDown / nb_runs
# aveknnPrecDown = aveknnPrecDown / nb_runs
# aveknnRecallDown = aveknnRecallDown / nb_runs
# avelrRecallUp = avelrRecallUp / nb_runs
# avelrPrecUp = avelrPrecUp / nb_runs
# avesvmRecallUp = avesvmRecallUp / nb_runs
# avesvmPrecUp = avesvmPrecUp / nb_runs
# aveknnRecallUp = aveknnRecallUp / nb_runs
# aveknnPrecUp = aveknnPrecUp / nb_runs
avelrIndices = avelrIndices / nb_runs
avelrTrainingError = avelrTrainingError / nb_runs
avelrTestingError = avelrTestingError / nb_runs
avesvmIndices = avesvmIndices / nb_runs
avesvmTrainingError = avesvmTrainingError / nb_runs
avesvmTestingError = avesvmTestingError / nb_runs
aveknnIndices = aveknnIndices / nb_runs
aveknnTrainingError = aveknnTrainingError / nb_runs
aveknnTestingError = aveknnTestingError / nb_runs


#Plotting averaged values:
# plotLines([[avelrRecallDown], [avelrPrecDown]], 'Averaged LR: Precision vs Recall(Down)', 'Recall', 'Precision')
# plotLines([[avesvmRecallDown], [avesvmPrecDown]], 'Averaged SVM: Precision vs Recall(Down)', 'Recall', 'Precision')
# plotLines([[aveknnRecallDown], [aveknnPrecDown]], 'Averaged kNN: Precision vs Recall(Down)', 'Recall', 'Precision')

# plotLines([[avelrRecallUp], [avelrPrecUp]], 'Averaged LR: Precision vs Recall(Up)', 'Recall', 'Precision')
# plotLines([[avesvmRecallUp], [avesvmPrecUp]], 'Averaged SVM: Precision vs Recall(Up)', 'Recall', 'Precision')
# plotLines([[aveknnRecallUp], [aveknnPrecUp]], 'Averaged kNN: Precision vs Recall(Up)', 'Recall', 'Precision')

plotLines([[avelrIndices, avelrIndices], [avelrTrainingError, avelrTestingError]], 'Averaged Errors:LogRegr', 'Dataset size', 'Error')
plotLines([[avesvmIndices, avesvmIndices], [avesvmTrainingError, avesvmTestingError]], 'Averaged Errors:SVM', 'Dataset size', 'Error')
plotLines([[aveknnIndices, aveknnIndices], [aveknnTrainingError, aveknnTestingError]], 'Averaged Errors:kNN', 'Dataset size', 'Error')