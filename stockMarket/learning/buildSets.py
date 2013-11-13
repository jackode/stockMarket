#-*- coding: utf-8 -*-

import scipy as sp
import numpy as np
import pickle as pi
from random import randint
from getData.models import TyroonStock

weather = sp.genfromtxt('../CSV Data/dbDump/weather_jorhat.csv', delimiter=';')
dweather = sp.genfromtxt(
    '../CSV Data/dbDump/dweather_jorhat.csv', delimiter=';')
ddweather = sp.genfromtxt(
    '../CSV Data/dbDump/ddweather_jorhat.csv', delimiter=';')
features35 = sp.genfromtxt(
    '../CSV Data/dbDump/features_tyroon.csv', delimiter=';')
dfeatures35 = sp.genfromtxt(
    '../CSV Data/dbDump/dfeatures_tyroon.csv', delimiter=';')
tyroon = sp.genfromtxt('../CSV Data/dbDump/stock_tyroon.csv', delimiter=';')
diana = sp.genfromtxt('../CSV Data/dbDump/stock_diana.csv', delimiter=';')
warrent = sp.genfromtxt('../CSV Data/dbDump/stock_warrent.csv', delimiter=';')
indian = sp.genfromtxt('../CSV Data/dbDump/stock_indian.csv', delimiter=';')
tenren = sp.genfromtxt('../CSV Data/dbDump/stock_tenren.csv', delimiter=';')

id = 0
temp = 1
hum = 2
wind = 3
press = 4
day = 5
month = 6
year = 7
dayStock = 1
monthStock = 2
yearStock = 3

dayFeatures = []
print 'Merging all CSV data into one variable...'
for row in diana:
    # Creates a fills dayFeatures
    today = row
    temp = weather[weather[:, day] == row[dayStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, month] == row[monthStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, year] == row[yearStock]]
    if len(temp) == 0:
        continue
    today = np.append(today, temp[0][1:5])

    temp = dweather[dweather[:, day] == row[dayStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, month] == row[monthStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, year] == row[yearStock]]
    if len(temp) == 0:
        continue
    today = np.append(today, temp[0][1:5])

    temp = ddweather[ddweather[:, day] == row[dayStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, month] == row[monthStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, year] == row[yearStock]]
    if len(temp) == 0:
        continue
    today = np.append(today, temp[0][1:5])

    temp = features35[features35[:, dayStock] == row[dayStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, monthStock] == row[monthStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, yearStock] == row[yearStock]]
    if len(temp) == 0:
        continue
    today = np.append(today, temp[0][4:35])

    temp = dfeatures35[dfeatures35[:, dayStock] == row[dayStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, monthStock] == row[monthStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, yearStock] == row[yearStock]]
    if len(temp) == 0:
        continue
    today = np.append(today, temp[0][4:35])

    temp = tyroon[tyroon[:, dayStock] == row[dayStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, monthStock] == row[monthStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, yearStock] == row[yearStock]]
    if len(temp) == 0:
        continue
    today = np.append(today, temp[0][4:10])

    temp = warrent[warrent[:, dayStock] == row[dayStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, monthStock] == row[monthStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, yearStock] == row[yearStock]]
    if len(temp) == 0:
        continue
    today = np.append(today, temp[0][4:10])

    temp = indian[indian[:, dayStock] == row[dayStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, monthStock] == row[monthStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, yearStock] == row[yearStock]]
    if len(temp) == 0:
        continue
    today = np.append(today, temp[0][4:10])

    temp = tenren[tenren[:, dayStock] == row[dayStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, monthStock] == row[monthStock]]
    if len(temp) == 0:
        continue
    temp = temp[temp[:, yearStock] == row[yearStock]]
    if len(temp) == 0:
        continue
    today = np.append(today, temp[0][4:10])

    dayFeatures.append(today)

dayFeatures = np.asarray(dayFeatures)

# Cleaning the data:
for i in xrange(len(dayFeatures[0])):
    if np.any(sp.isnan(dayFeatures[:, i])):
        print 'Cleaning column:' + str(i)
        dayFeatures = dayFeatures[~sp.isnan(dayFeatures[:, i])]

print 'dayFeatures now contains ' + str(len(dayFeatures)) + ' entries.'

# NEXT: Divide it into  sets. training: 70%, cv(for rfe
# nbFeaturesToSelect): 20%, test(real score): 10%
featLen = len(dayFeatures)
testLen = int(0.1 * featLen)
cvLen = int(0.2 * featLen)

testIds = []
cvIds = []
print 'Generating Cross-Validation and Test sets...'
for i in xrange(testLen):
    testIds.append(randint(260, featLen - 1))

for i in xrange(cvLen):
    cvIds.append(randint(260, featLen - 1))

testIds = np.sort(testIds)
cvIds = np.sort(cvIds)

# Get the "answers" for when we'll learn
print 'Generating the wanted results...'
finalAnswers = []
testAnswers = []
cvAnswers = []
allAnswers = []
for i in xrange(260, len(dayFeatures)):
    old = TyroonStock.objects.get(
        day=dayFeatures[i, dayStock], month=dayFeatures[i, monthStock], year=dayFeatures[i, yearStock])
    curr = TyroonStock.objects.get(id=old.id - 1)
    answer = 0 if curr.close > old.close else 1
    allAnswers.append(answer)
    if i in cvIds:
        cvAnswers.append(answer)
    elif i in testIds:
        testAnswers.append(answer)
    else:
        finalAnswers.append(answer)




# Saves dayFeatures in file (Should have length of 1135)
# Read dayFeatures from file:
# file = open('p.pickle', 'rb')
# pi.load(file)
print 'Saving Variables...'

file = open('dayFeatures.pi', 'wb')
pi.dump(dayFeatures.tolist(), file)
file = open('cvIds.pi', 'wb')
pi.dump(cvIds, file)
file = open('testIds.pi', 'wb')
pi.dump(testIds, file)
file = open('allAnswers.pi', 'wb')
pi.dump(allAnswers, file)
file = open('cvAnswers.pi', 'wb')
pi.dump(cvAnswers, file)
# file = open('finalAnswers.pi', 'wb')
# pi.dump(finalAnswers, file)
# file = open('testAnswers.pi', 'wb')
# pi.dump(testAnswers, file)