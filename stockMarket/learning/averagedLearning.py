#-*- coding: utf-8 -*-

import os
import pickle as pi
import scipy as sp
import numpy as np


curr_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
execfile(curr_dir + '/../Scripts/plot.py')

def buildSet():
    execfile(curr_dir + '/learning/buildSets.py')

def learnFromSet():
    execfile(curr_dir + '/learning/learning.py')



file = open('lrPrecDown', 'rb')
avelrPrecDown = pi.load(file)

file = open('lrRecallDown', 'rb')
avelrRecallDown = pi.load(file)

file = open('svmPrecDown', 'rb')
avesvmPrecDown = pi.load(file)

file = open('svmRecallDown', 'rb')
avesvmRecallDown = pi.load(file)

file = open('knnPrecDown', 'rb')
aveknnPrecDown = pi.load(file)

file = open('knnRecallDown', 'rb')
aveknnRecallDown = pi.load(file)

file = open('lrRecallUp', 'rb')
avelrRecallUp = pi.load(file)

file = open('lrPrecUp', 'rb')
avelrPrecUp = pi.load(file)

file = open('svmRecallUp', 'rb')
avesvmRecallUp = pi.load(file)

file = open('svmPrecUp', 'rb')
avesvmPrecUp = pi.load(file)

file = open('knnRecallUp', 'rb')
aveknnRecallUp = pi.load(file)

file = open('knnPrecUp', 'rb')
aveknnPrecUp = pi.load(file)

file = open('lrIndices', 'rb')
avelrIndices = pi.load(file)

file = open('lrTrainingError', 'rb')
avelrTrainingError = pi.load(file)

file = open('lrTestingError', 'rb')
avelrTestingError = pi.load(file)

file = open('svmIndices', 'rb')
avesvmIndices = pi.load(file)

file = open('svmTrainingError', 'rb')
avesvmTrainingError = pi.load(file)

file = open('svmTestingError', 'rb')
avesvmTestingError = pi.load(file)

file = open('knnIndices', 'rb')
aveknnIndices = pi.load(file)

file = open('knnTrainingError', 'rb')
aveknnTrainingError = pi.load(file)

file = open('knnTestingError', 'rb')
aveknnTestingError = pi.load(file)

nb_runs = 3

for i in xrange(1, nb_runs):
    buildSet()
    learnFromSet()

    file = open('lrPrecDown', 'rb')
    lrPrecDown = pi.load(file)
    avelrPrecDown += lrPrecDown

    file = open('lrRecallDown', 'rb')
    lrRecallDown = pi.load(file)
    avelrRecallDown += lrRecallDown

    file = open('svmPrecDown', 'rb')
    svmPrecDown = pi.load(file)
    avesvmPrecDown += svmPrecDown

    file = open('svmRecallDown', 'rb')
    svmRecallDown = pi.load(file)
    avesvmRecallDown += svmRecallDown

    file = open('knnPrecDown', 'rb')
    knnPrecDown = pi.load(file)
    aveknnPrecDown += knnPrecDown

    file = open('knnRecallDown', 'rb')
    knnRecallDown = pi.load(file)
    aveknnRecallDown += knnRecallDown

    file = open('lrRecallUp', 'rb')
    lrRecallUp = pi.load(file)
    avelrRecallUp += lrRecallUp

    file = open('lrPrecUp', 'rb')
    lrPrecUp = pi.load(file)
    avelrPrecUp += lrPrecUp

    file = open('svmRecallUp', 'rb')
    svmRecallUp = pi.load(file)
    avesvmRecallUp += svmRecallUp

    file = open('svmPrecUp', 'rb')
    svmPrecUp = pi.load(file)
    avesvmPrecUp += svmPrecUp

    file = open('knnRecallUp', 'rb')
    knnRecallUp = pi.load(file)
    aveknnRecallUp += knnRecallUp

    file = open('knnPrecUp', 'rb')
    knnPrecUp = pi.load(file)
    aveknnPrecUp += knnPrecUp

    file = open('lrIndices', 'rb')
    lrIndices = pi.load(file)
    avelrIndices += lrIndices

    file = open('lrTrainingError', 'rb')
    lrTrainingError = pi.load(file)
    avelrTrainingError += lrTrainingError

    file = open('lrTestingError', 'rb')
    lrTestingError = pi.load(file)
    avelrTestingError += lrTestingError

    file = open('svmIndices', 'rb')
    svmIndices = pi.load(file)
    avesvmIndices += svmIndices

    file = open('svmTrainingError', 'rb')
    svmTrainingError = pi.load(file)
    avesvmTrainingError += svmTrainingError

    file = open('svmTestingError', 'rb')
    svmTestingError = pi.load(file)
    avesvmTestingError += svmTestingError

    file = open('knnIndices', 'rb')
    knnIndices = pi.load(file)
    aveknnIndices += knnIndices

    file = open('knnTrainingError', 'rb')
    knnTrainingError = pi.load(file)
    aveknnTrainingError += knnTrainingError

    file = open('knnTestingError', 'rb')
    knnTestingError = pi.load(file)
    aveknnTestingError += knnTestingError


#Averaging Everything:
avelrPrecDown = avelrPrecDown / nb_runs
avelrRecallDown = avelrRecallDown / nb_runs
avesvmPrecDown = avesvmPrecDown / nb_runs
avesvmRecallDown = avesvmRecallDown / nb_runs
aveknnPrecDown = aveknnPrecDown / nb_runs
aveknnRecallDown = aveknnRecallDown / nb_runs
avelrRecallUp = avelrRecallUp / nb_runs
avelrPrecUp = avelrPrecUp / nb_runs
avesvmRecallUp = avesvmRecallUp / nb_runs
avesvmPrecUp = avesvmPrecUp / nb_runs
aveknnRecallUp = aveknnRecallUp / nb_runs
aveknnPrecUp = aveknnPrecUp / nb_runs
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
plotLines([[avelrRecallDown], [avelrPrecDown]], 'Averaged LR: Precision vs Recall(Down)', 'Recall', 'Precision')
plotLines([[avesvmRecallDown], [avesvmPrecDown]], 'Averaged SVM: Precision vs Recall(Down)', 'Recall', 'Precision')
plotLines([[aveknnRecallDown], [aveknnPrecDown]], 'Averaged kNN: Precision vs Recall(Down)', 'Recall', 'Precision')

plotLines([[avelrRecallUp], [avelrPrecUp]], 'Averaged LR: Precision vs Recall(Up)', 'Recall', 'Precision')
plotLines([[avesvmRecallUp], [avesvmPrecUp]], 'Averaged SVM: Precision vs Recall(Up)', 'Recall', 'Precision')
plotLines([[aveknnRecallUp], [aveknnPrecUp]], 'Averaged kNN: Precision vs Recall(Up)', 'Recall', 'Precision')

plotLines([[avelrIndices, avelrIndices], [avelrTrainingError, avelrTestingError]], 'Averaged Errors:LogRegr', 'Dataset size', 'Error')
plotLines([[avesvmIndices, avesvmIndices], [avesvmTrainingError, avesvmTestingError]], 'Averaged Errors:SVM', 'Dataset size', 'Error')
plotLines([[aveknnIndices, aveknnIndices], [aveknnTrainingError, aveknnTestingError]], 'Averaged Errors:kNN', 'Dataset size', 'Error')