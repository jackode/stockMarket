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

#Gets content of p.pickle, performs the learning task and evaluates itself
def openFiles():
    print 'Openning files...'
    file = open('dayFeatures.pi', 'rb')
    dayFeatures = np.asarray(pi.load(file))
    file = open('cvIds.pi', 'rb')
    cvIds = pi.load(file)
    file = open('testIds.pi', 'rb')
    testIds = pi.load(file)
    file = open('allAnswers.pi', 'rb')
    allAnswers = pi.load(file)
    return (dayFeatures, cvIds, testIds, allAnswers)

#Scaling the features:
def scaleFeatures(dayFeatures):
    print 'Scaling features...'
    for i in xrange(len(dayFeatures[0])):
        dayFeatures[:, i] = (dayFeatures[:, i] - np.average(dayFeatures[:, i])) / (np.amax(dayFeatures[:, i]) - np.amin(dayFeatures[:, i]))
    return dayFeatures

#Building features inputs
def buildFeaturesInputs(dayFeatures):
    allFeatures = []
    print 'Building features inputs...'
    for i in xrange(260, len(dayFeatures)):
        feature = np.array([])
        for j in xrange(0, 260):
            feature = np.append(feature, dayFeatures[i-j])
        allFeatures.append(feature)
    return allFeatures

#Building Sets:
def buildSets(allFeatures, allAnswers):
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
    return (finalFeatures, finalAnswers, cvFeatures, cvAnswers, testFeatures, testAnswers)

#Shuffling Training Features:
def shuffleSet(finalFeatures, finalAnswers):
    shuffled = np.asarray(finalFeatures)
    shuffled = np.hstack((shuffled, np.zeros((shuffled.shape[0], 1), dtype=shuffled.dtype)))
    shuffled[:, -1] = finalAnswers
    np.random.shuffle(shuffled)
    finalFeatures = shuffled[:, :-1]
    finalAnswers = shuffled[:, -1]
    return (finalFeatures, finalAnswers)

#Applying LDA:
def applyLDA(finalFeatures, finalAnswers, cvFeatures, testFeatures, n_components=200):
    print 'Applying LDA...'
    selLda = lda.LDA(n_components=n_components)
    finalFeatures = selLda.fit_transform(finalFeatures, finalAnswers)
    cvFeatures = selLda.transform(cvFeatures)
    testFeatures = selLda.transform(testFeatures)
    return (finalFeatures, cvFeatures, testFeatures)

#Applying PCA:
def applyPCA(finalFeatures, cvFeatures, testFeatures, n_components=2000):
    print 'Applying PCA...'
    pca = decomposition.PCA(n_components=n_components)
    finalFeatures = pca.fit_transform(finalFeatures)
    cvFeatures = pca.transform(cvFeatures)
    testFeatures = pca.transform(testFeatures)
    print 'With ' + str(n_components) + ' components we have a conserved variance of ' + str(np.average(pca.explained_variance_ratio_))
    return (finalFeatures, cvFeatures, testFeatures)


##########################
# Plotting Training and Test errors:
##########################

#Calculating Trainer Error:
def findTrainerError(trainer, trainer_name, finalFeatures, finalAnswers, testFeatures, testAnswers):
    print 'Calculating ' + trainer_name + ' Errors...'
    start = 10
    trainingError = []
    testingError = []
    indices = []
    for i in xrange(start, len(finalFeatures)):
        trainer.fit(finalFeatures[0:i+1], finalAnswers[0:i+1])
        error = trainer.score(finalFeatures[0:i+1], finalAnswers[0:i+1])
        trainingError.append(1-error)
        indices.append(i)
        error = trainer.score(testFeatures, testAnswers)
        testingError.append(1-error)
    print 'Final ' + trainer_name + ' score is ' + str(trainer.score(testFeatures, testAnswers))
    return (indices, trainingError, testingError)

#Only train and get score of algorithm:
def trainerLearnScore(trainer, trainer_name, finalFeatures, finalAnswers, testFeatures, testAnswers):
    print 'Training ' + trainer_name + '...'
    trainer.fit(finalFeatures, finalAnswers)
    score = trainer.score(testFeatures, testAnswers)
    print 'Final ' + trainer_name + ' score is ' + score

#Plot Training and Post Errors
def plotErrors(trainer_name, indices, trainError, testError):
    print 'Plotting error results...'
    plotLines([[indices, indices], [trainError, testError]], 'Errors:' + trainer_name, 'Dataset size', 'Error')

#Plot and return Precision and Recall
def plotPrecisionRecall(learner, learner_name, testFeatures, testAnswers):
    print 'Plotting Precision and Recall for ' + learner_name
    precDown, recDown, thrDown = precision_recall_curve(testAnswers == 0, learner.predict_proba(testFeatures)[:, 0])
    precUp, recUp, thrUp = precision_recall_curve(testAnswers == 1, learner.predict_proba(testFeatures)[:, 1])
    plotLines([[recDown], [precDown]], learner_name + ': Precision vs Recall(Down)', 'Recall', 'Precision')
    plotLines([[recUp], [precUp]], learner_name + ': Precision vs Recall(Up)', 'Recall', 'Precision')
    return (precDown, recDown, thrDown, precUp, recUp, thrUp)

# Printing Classification Report:
def printClassReport(trainer, trainer_name, testAnswers, testFeatures, thrUp, thrDown):
    print 'Report for ' + trainer_name + ' (Up):'
    print classification_report(testAnswers == 1, trainer.predict_proba(testFeatures)[:, 1] > thrUp, target_names=['Don\'t know', 'Will go up'])
    print 'Report for ' + trainer_name + ' (Down):'
    print classification_report(testAnswers == 0, trainer.predict_proba(testFeatures)[:, 0] > thrDown, target_names=['Don\'t know', 'Will go Down'])

#Exporting as files
def exportResults(lrPrecDown, lrRecDown, svmPrecDown, svmRecDown, knnPrecDown, knnRecDown, lrRecUp, lrPrecUp, svmRecUp, svmPrecUp, knnRecUp, knnPrecUp, lrIndices, lrTrainingError, lrTestingError, svmIndices, svmTrainingError, svmTestingError, knnIndices, knnTrainingError, knnTestingError, cvAnswers):
    print 'Exporting as files...'
    # file = open('lrPrecDown', 'wb')
    # pi.dump(lrPrecDown, file)
    # file = open('lrRecDown', 'wb')
    # pi.dump(lrRecDown, file)
    # file = open('svmPrecDown', 'wb')
    # pi.dump(svmPrecDown, file)
    # file = open('svmRecDown', 'wb')
    # pi.dump(svmRecDown, file)
    # file = open('knnPrecDown', 'wb')
    # pi.dump(knnPrecDown, file)
    # file = open('knnRecDown', 'wb')
    # pi.dump(knnRecDown, file)
    # file = open('lrRecUp', 'wb')
    # pi.dump(lrRecUp, file)
    # file = open('lrPrecUp', 'wb')
    # pi.dump(lrPrecUp, file)
    # file = open('svmRecUp', 'wb')
    # pi.dump(svmRecUp, file)
    # file = open('svmPrecUp', 'wb')
    # pi.dump(svmPrecUp, file)
    # file = open('knnRecUp', 'wb')
    # pi.dump(knnRecUp, file)
    # file = open('knnPrecUp', 'wb')
    # pi.dump(knnPrecUp, file)
    
    file = open('lrIndices', 'wb')
    pi.dump(lrIndices, file)
    file = open('lrTrainingError', 'wb')
    pi.dump(lrTrainingError, file)
    file = open('lrTestingError', 'wb')
    pi.dump(lrTestingError, file)
    file = open('svmIndices', 'wb')
    pi.dump(svmIndices, file)
    file = open('svmTrainingError', 'wb')
    pi.dump(svmTrainingError, file)
    file = open('svmTestingError', 'wb')
    pi.dump(svmTestingError, file)
    file = open('knnIndices', 'wb')
    pi.dump(knnIndices, file)
    file = open('knnTrainingError', 'wb')
    pi.dump(knnTrainingError, file)
    file = open('knnTestingError', 'wb')
    pi.dump(knnTestingError, file)
    file = open('cvAnswers.pi', 'wb')
    pi.dump(cvAnswers, file)


dayFeatures, cvIds, testIds, allAnswers = openFiles()
dayFeatures = scaleFeatures(dayFeatures)
allFeatures = buildFeaturesInputs(dayFeatures)
finalFeatures, finalAnswers, cvFeatures, cvAnswers, testFeatures, testAnswers = buildSets(allFeatures, allAnswers)
finalFeatures, finalAnswers = shuffleSet(finalFeatures, finalAnswers)
finalFeatures, cvFeatures, testFeatures = applyLDA(finalFeatures, finalAnswers, cvFeatures, testFeatures, 200)
# finalFeatures, cvFeatures, testFeatures = applyPCA(finalFeatures, cvFeatures, testFeatures, n_components=2000)

#Convert features and answers into arrays:
testFeatures = np.asarray(testFeatures)
testAnswers = np.asarray(testAnswers)
finalFeatures = np.asarray(finalFeatures)
finalAnswers = np.asarray(finalAnswers)
cvFeatures = np.asarray(cvFeatures)
cvAnswers = np.asarray(cvAnswers)

#Creation of the learners:
lrLearner = LogisticRegression(penalty='l2', dual=False, C=1.0)
svmLearner = svm.SVC(C=250.0, kernel="rbf", probability=True)
knnLearner = neighbors.KNeighborsClassifier(n_neighbors=350, algorithm='auto')

#Finding Errors:
lrTrainingError, lrTestingError, lrIndices = findTrainerError(lrLearner, 'LogReg', finalFeatures, finalAnswers, testFeatures, testAnswers)
svmTrainingError, svmTestingError, svmIndices = findTrainerError(svmLearner, 'SVM', finalFeatures, finalAnswers, testFeatures, testAnswers)
knnTrainingError, knnTestingError, knnIndices = findTrainerError(knnLearner, 'kNN', finalFeatures, finalAnswers, testFeatures, testAnswers)

#Executing plot files to have additional functions
curr_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
execfile(curr_dir + '/../Scripts/plot.py')

#Plot and get Precision and Recall:
lrPrecDown, lrRecDown, lrThrDown, lrPrecUp, lrRecUp, lrThrUp = plotPrecisionRecall(lrLearner, 'LogReg', testFeatures, testAnswers)
svmPrecDown, svmRecDown, svmThrDown, svmPrecUp, svmRecUp, svmThrUp = plotPrecisionRecall(svmLearner, 'SVM', testFeatures, testAnswers)
knnPrecDown, knnRecDown, knnThrDown, knnPrecUp, knnRecUp, knnThrUp = plotPrecisionRecall(knnLearner, 'kNN', testFeatures, testAnswers)

#Print Classification Report:
printClassReport(lrTrainer, 'LogReg', testAnswers, testFeatures, 0.54, 0.008)
printClassReport(lrTrainer, 'SVM', testAnswers, testFeatures, 0.54, 0.05)
printClassReport(lrTrainer, 'kNN', testAnswers, testFeatures, 0.51, 0.3)

#Exporting values in files
exportResults(lrPrecDown, lrRecDown, svmPrecDown, svmRecDown, knnPrecDown, knnRecDown, lrRecUp, lrPrecUp, svmRecUp, svmPrecUp, knnRecUp, knnPrecUp, lrIndices, lrTrainingError, lrTestingError, svmIndices, svmTrainingError, svmTestingError, knnIndices, knnTrainingError, knnTestingError, cvAnswers)

