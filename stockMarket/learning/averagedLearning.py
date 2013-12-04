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
from multiprocessing import Process, Manager


curr_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
execfile(curr_dir + '/../Scripts/plot.py')


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
def buildSets(allFeatures, allAnswers, cvIds, testIds):
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
    print 'With ' + str(n_components) + ' components we have a conserved variance of ' + str(sp.sum(pca.explained_variance_ratio_))
    return (finalFeatures, cvFeatures, testFeatures)


##########################
# Plotting Training and Test errors:
##########################

#Calculating Trainer Error:
def findTrainerError(trainer, trainer_name, finalFeatures, finalAnswers, testFeatures, testAnswers, trainingError, testingError, indices):
    print 'Calculating ' + trainer_name + ' Errors...'
    start = 10
    # trainingError = []
    # testingError = []
    # indices = []
    ns = trainer
    trainer = ns.learner
    for i in xrange(start, len(finalFeatures)):
        trainer.fit(finalFeatures[0:i+1], finalAnswers[0:i+1])
        error = trainer.score(finalFeatures[0:i+1], finalAnswers[0:i+1])
        trainingError.append(1-error)
        indices.append(i)
        error = trainer.score(testFeatures, testAnswers)
        testingError.append(1-error)
    print 'Final ' + trainer_name + ' score is ' + str(trainer.score(testFeatures, testAnswers))
    ns.learner = trainer
    trainer = ns
    return (indices, trainingError, testingError)

#Finding Errors (in parallel):
def findTrainerErrorParallel(lrLearner, svmLearner, knnLearner, finalFeatures, finalAnswers, testFeatures, testAnswers):
    manager = Manager()
    lrTrainingError, lrTestingError, lrIndices = manager.list(), manager.list(), manager.list()
    svmTrainingError, svmTestingError, svmIndices = manager.list(), manager.list(), manager.list()
    knnTrainingError, knnTestingError, knnIndices = manager.list(), manager.list(), manager.list()

    temp = manager.Namespace()
    temp.learner = lrLearner
    lrLearner = temp

    temp = manager.Namespace()
    temp.learner = svmLearner
    svmLearner = temp

    temp = manager.Namespace()
    temp.learner = knnLearner
    knnLearner = temp

    lrP = Process(target=findTrainerError, args=(lrLearner, 'LogReg', finalFeatures, finalAnswers, testFeatures, testAnswers, lrTrainingError, lrTestingError, lrIndices))
    svmP = Process(target=findTrainerError, args=(svmLearner, 'SVM', finalFeatures, finalAnswers, testFeatures, testAnswers, svmTrainingError, svmTestingError, svmIndices))
    knnP = Process(target=findTrainerError, args=(knnLearner, 'kNN', finalFeatures, finalAnswers, testFeatures, testAnswers, knnTrainingError, knnTestingError, knnIndices))

    lrP.start()
    svmP.start()
    knnP.start()

    lrP.join()
    svmP.join()
    knnP.join()

    lrLearner = lrLearner.learner
    svmLearner = svmLearner.learner
    knnLearner = knnLearner.learner
    lrTrainingError, lrTestingError, lrIndices = np.asarray(lrTrainingError), np.asarray(lrTestingError), np.asarray(lrIndices)
    svmTrainingError, svmTestingError, svmIndices = np.asarray(svmTrainingError), np.asarray(svmTestingError), np.asarray(svmIndices)
    knnTrainingError, knnTestingError, knnIndices = np.asarray(knnTrainingError), np.asarray(knnTestingError), np.asarray(knnIndices)
    return (lrLearner, svmLearner, knnLearner, lrTrainingError, lrTestingError, lrIndices, svmTrainingError, svmTestingError, svmIndices, knnTrainingError, knnTestingError, knnIndices)

#Only train and get score of algorithm:
def trainerLearnScore(trainer, trainer_name, finalFeatures, finalAnswers, testFeatures, testAnswers, score):
    print 'Training ' + trainer_name + '...'
    ns = trainer
    trainer = ns.learner
    trainer.fit(finalFeatures, finalAnswers)
    score = trainer.score(testFeatures, testAnswers)
    print 'Final ' + trainer_name + ' score is ' + str(score)
    ns.learner = trainer
    trainer = ns
    return trainer

#Only train and get score of algorithm (Parallel):
def trainerLearnScoreParallel(lrLearner, svmLearner, knnLearner):
    manager = Manager()

    lrScore, svmScore, knnScore = manager.Value('d', 0.0), manager.Value('d', 0.0), manager.Value('d', 0.0)

    temp = manager.Namespace()
    temp.learner = lrLearner
    lrLearner = temp

    temp = manager.Namespace()
    temp.learner = svmLearner
    svmLearner = temp

    temp = manager.Namespace()
    temp.learner = knnLearner
    knnLearner = temp

    lrP = Process(target=trainerLearnScore, args=(lrLearner, 'LogReg', finalFeatures, finalAnswers, testFeatures, testAnswers, lrScore))
    svmP = Process(target=trainerLearnScore, args=(svmLearner, 'SVM', finalFeatures, finalAnswers, testFeatures, testAnswers, svmScore))
    knnP = Process(target=trainerLearnScore, args=(knnLearner, 'kNN', finalFeatures, finalAnswers, testFeatures, testAnswers, knnScore))

    lrP.start()
    svmP.start()
    knnP.start()

    lrP.join()
    svmP.join()
    knnP.join()

    lrLearner = lrLearner.learner
    svmLearner = svmLearner.learner
    knnLearner = knnLearner.learner
    return (lrLearner, svmLearner, knnLearner, lrScore, svmScore, knnScore)

#Plot Training and Post Errors
def plotErrors(trainer_name, indices, trainError, testError):
    print 'Plotting ' + trainer_name + ' error results...'
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



def buildSet():
    execfile(curr_dir + '/learning/buildSets.py')

def learnFromSet():
    #Build and Work on features
    dayFeatures, cvIds, testIds, allAnswers = openFiles()
    dayFeatures = scaleFeatures(dayFeatures)
    allFeatures = buildFeaturesInputs(dayFeatures)
    finalFeatures, finalAnswers, cvFeatures, cvAnswers, testFeatures, testAnswers = buildSets(allFeatures, allAnswers, cvIds, testIds)
    finalFeatures, finalAnswers = shuffleSet(finalFeatures, finalAnswers)
    # finalFeatures, cvFeatures, testFeatures = applyLDA(finalFeatures, finalAnswers, cvFeatures, testFeatures, 2000)
    finalFeatures, cvFeatures, testFeatures = applyPCA(finalFeatures, cvFeatures, testFeatures, n_components=1000)

    #Convert features and answers into arrays:
    testFeatures = np.asarray(testFeatures)
    testAnswers = np.asarray(testAnswers)
    finalFeatures = np.asarray(finalFeatures)
    finalAnswers = np.asarray(finalAnswers)
    cvFeatures = np.asarray(cvFeatures)
    cvAnswers = np.asarray(cvAnswers)

    #Creation of the learners:
    lrLearner = LogisticRegression(penalty='l1', dual=False, C=1000.0)
    svmLearner = svm.SVC(C=5, kernel='poly', degree=4)
    knnLearner = neighbors.KNeighborsClassifier(n_neighbors=350, algorithm='auto')

    return findTrainerErrorParallel(lrLearner, svmLearner, knnLearner, finalFeatures, finalAnswers, testFeatures, testAnswers)



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

nb_runs = 18

for i in xrange(1, nb_runs):
    print i
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

plotLines([[avelrIndices, avelrIndices], [avelrTrainingError, avelrTestingError]], 'Averaged No Weather Errors:LogRegr', 'Dataset size', 'Error')
plotLines([[avesvmIndices, avesvmIndices], [avesvmTrainingError, avesvmTestingError]], 'Averaged No Weather Errors:SVM', 'Dataset size', 'Error')
plotLines([[aveknnIndices, aveknnIndices], [aveknnTrainingError, aveknnTestingError]], 'Averaged No Weather Errors:kNN', 'Dataset size', 'Error')