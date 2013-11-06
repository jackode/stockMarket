#-*- coding: utf-8 -*-
from getData.models import *
import scipy as sp
import numpy as np

# db = TyroonStock
# tyroon = db.objects.all()
tyroon = sp.genfromtxt('../CSV Data/dbDump/stock_diana.csv', delimiter=';')
N = 14
day, month, year, open, close, low, high, adj, volume = 1, 2, 3, 4, 5, 6, 7, 8, 9
i = 0
previousEMA = 0.0
previousLow = previousHigh = extremePoint = previousSar = previousAF = previousATR = previousADL = Low = High = 0.0

# Testing only:
# Also, CHANGE DIANA WITH TYROON !
tyroon = tyroon[:50]
#

def mean_abs_dev(pop):
    alksjdhfalksd = float ( len (pop))
    alskdjfhladsasdfjasf  = sum (pop) / alksjdhfalksd
    qewrnqwerqnwer = [ abs (x - alskdjfhladsasdfjasf ) for x in pop ]
    return sum (qewrnqwerqnwer) / alksjdhfalksd


tyroon = tyroon[::-1]
for t in tyroon:
    entry = Feature35(
        day=t[day],
        month=t[month],
        year=t[year]
    )
    if i % N == 0:
        previousLow = Low
        previousHigh = High
        Low = np.amin(tyroon[i - N:i, close]) if i != 0 else 0
        High = np.amax(tyroon[i - N:i, close]) if i != 0 else 0

    # Exponential Moving Average:
    smoothing = 2.0 / 11.0
    previousEMA = ((t[close] - previousEMA) * smoothing) + previousEMA
    entry.movAverageExp = previousEMA

    #%B Indicator
    divider = t[high] - t[low] if t[high] - t[low] != 0 else 1
    entry.indicB = (t[close] - t[low]) / divider

    if i >= 1:
        #Ease of Movement
        if High != 0 and Low != 0 and t[volume] != 0:
            entry.easeMove = ((High + Low)/2 - (previousHigh + previousLow)/2) / ((t[volume]/100000)/(High - Low))
        else:
            entry.easeMove = 0

        #Force Index
        entry.forceIndex = (t[close] - tyroon[i-1, close]) * t[volume]

        # Price Volume Trend:
        entry.priceVolumeTrend = t[
            volume] * (t[close] - tyroon[i - 1][close]) / tyroon[i - 1][close]

    if i >= 5:
        # 5 day disparity:
        temp = np.mean(tyroon[i - 5:i, close])
        entry.day5disparity = (t[close] / temp) * 100

    if i >= 10:
        # 10 day disparity:
        temp = np.mean(tyroon[i - 10:i, close])
        entry.day10disparity = (t[close] / temp) * 100

        #MACD (Custom)
        entry.macd = entry.day5disparity - entry.day5disparity - entry.movAverageExp

        #Comodity Channel Index:
        typical = (High + Low + t[close]) / 3
        meanDeviation = mean_abs_dev(tyroon[i-10:i, close])
        entry.commChanIndex = (typical - entry.day10disparity) / (0.015 * meanDeviation)


    if i >= N:

        #Money Flow Index
        negMFlow = []
        posMFlow = []
        for j in xrange(N-1):
            if tyroon[i-N+j, close] > tyroon[i-N+j+1, close]:
                negMFlow.append(tyroon[i-N+j+1])
            else:
                posMFlow.append(tyroon[i-N+j+1])
        negMFlow = np.asarray(negMFlow)
        posMFlow = np.asarray(posMFlow)
        negMFlow = sp.sum(negMFlow[:, volume])*(np.average(negMFlow[:, close]) + High + Low)/3 + 1
        posMFlow = sp.sum(posMFlow[:, volume])*(np.average(posMFlow[:, close]) + High + Low)/3 + 1
        mFlowRatio = posMFlow/negMFlow
        entry.monneyFI = 100 - (100/(1 + mFlowRatio))

        #Momentum:
        entry.momentum = (t[close] / tyroon[i - N][close]) * 100

        #Stochastik % K
        entry.stochK = 100 * (t[close] - Low) / (High - Low)

        #Parabolic SAR
        if extremePoint < np.amax(tyroon[i - N, high]):
            extremePoint = np.amax(tyroon[i - N, high])
            previousAF = previousAF + \
                0.02 if (previousAF + 0.02) <= 0.2 else 0.2
        if i == 0:
            previousSar = t[low]
        entry.paraSar = previousSar

        # Accumulation Distribution Line:
        mFlowMultplier = ((t[close] - Low) - (High - t[close])) / (High - Low)
        mFlowVolume = mFlowMultplier * sp.sum(tyroon[i-N:i, volume])
        entry.accDistrLine = previousADL

        #Chaikin Money Flow:
        mFlowVolume = mFlowMultplier * t[volume]
        entry.chaikinMF = mFlowVolume / sp.sum(tyroon[i-N:i, volume])

        #Detrended Price Oscillator
        entry.detrPriceOsc = tyroon[i - ((N/2)+1), close] - entry.day10disparity

        #Average True Range:
        trueRange = max(High - Low, High - tyroon[i-1, close], Low - tyroon[i-1, close])
        entry.avTrueRange = previousATR


        #updating the time period:
        if i % N == 0:
            previousATR = ((previousATR * 13) + trueRange)/14
            previousADL = previousADL + mFlowVolume
            previousSar = previousSar + \
                previousAF * (extremePoint - previousSar)


    entry.save()
    i = i + 1
