#-*- coding: utf-8 -*-
from getData.models import *
import scipy as sp
import numpy as np
import pdb

tyroon = sp.genfromtxt('../CSV Data/dbDump/stock_tyroon.csv', delimiter=';')
N = 14
day, month, year, open, close, low, high, adj, volume = 1, 2, 3, 4, 5, 6, 7, 8, 9
i = 0
previousEMA = 0.0
standardDeviation = volNEMA = vol2NEMA = previousLow = previousHigh = extremePoint = previousSar = previousAF = previousATR = previousADL = Low = High = 0.0
cumulativeNVI = 1000
previousAG = previousAL = 1
minRSI = float('inf')
maxRSI = float('-inf')


def mean_abs_dev(pop):
    alksjdhfalksd = float(len(pop))
    alskdjfhladsasdfjasf = sum(pop) / alksjdhfalksd
    qewrnqwerqnwer = [abs(x - alskdjfhladsasdfjasf) for x in pop]
    return sum(qewrnqwerqnwer) / alksjdhfalksd

tyroon = tyroon[:50]
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
        Low = np.amin(tyroon[i - N:i + 1, high]) if i != 0 else 0
        High = np.amax(tyroon[i - N:i + 1, low]) if i != 0 else 0
        if i != 0:
            aveG = len(filter(None, [obj > tyroon[i - N, close] and obj for obj in tyroon[i - N:i + 1, close]]))
            aveL = len(filter(None, [obj < tyroon[i - N, close] and obj for obj in tyroon[i - N:i + 1, close]]))
            if aveL != 0 and aveG != 0:
                previousAG =  np.average(filter(None, [obj > tyroon[i - N, close] and obj for obj in tyroon[i - N:i + 1, close]]))/ N if i == N else ((N - 1) * previousAG + np.average(filter(None, [obj > tyroon[i - N, close] and obj for obj in tyroon[i - N:i + 1, close]]))) / N
                previousAL = np.average(filter(None, [obj < tyroon[i - N, close] and obj for obj in tyroon[i - N:i + 1, close]])) / N if i == N else ((N - 1) * previousAG + np.average(filter(None, [obj < tyroon[i - N, close] and obj for obj in tyroon[i - N:i + 1, close]]))) / N
            else:
                previousAL = previousAG = 0

    # Relative Strength Index:
    entry.relStrengthI = 100 - (100 / (1 + (previousAG / previousAL))) if previousAL != 0 else 0
    minRSI = min(minRSI, entry.relStrengthI)
    maxRSI = max(maxRSI, entry.relStrengthI)

    # Stochastic RSI:
    if minRSI != maxRSI:
        entry.stochRSI = (entry.relStrengthI - minRSI) / (maxRSI - minRSI)
    else:
        entry.stochRSI = 0

    # Stochastic Oscillator:
    if np.amax(tyroon[0:i + 1, high]) != np.amin(tyroon[0:i + 1, low]):
        entry.stochOsc = 100 * (t[close] - np.amin(tyroon[0:i + 1, low])) / (np.amax(tyroon[0:i + 1, high]) - np.amin(tyroon[0:i + 1, low]))
    else:
        entry.stochOsc = 0

    # Price Relative for each Concurrent:
    try:
        entry.priceRelDiana = t[close] / float(
            DianaStock.objects.get(day=t[day], month=t[month], year=t[year]).close)
    except:
        entry.priceRelDiana = 0

    try:
        entry.priceRelWarrent = t[close] / float(
            WarrentStock.objects.get(day=t[day], month=t[month], year=t[year]).close)
    except:
        entry.priceRelWarrent = 0

    try:
        entry.priceRelTenren = t[close] / float(
            TenRenStock.objects.get(day=t[day], month=t[month], year=t[year]).close)
    except:
        entry.priceRelTenren = 0

    try:
        entry.priceRelAsian = t[close] / float(
            IndianStock.objects.get(day=t[day], month=t[month], year=t[year]).close)
    except:
        entry.priceRelAsian = 0

    # Exponential Moving Average:
    smoothing = 2.0 / 11.0
    previousEMA = ((t[close] - previousEMA) * smoothing) + previousEMA
    entry.movAverageExp = previousEMA

    #%B Indicator
    divider = t[high] - t[low] if t[high] - t[low] != 0 else 1
    entry.indicB = (t[close] - t[low]) / divider

    if i >= 1:
        # William %R:
        if np.amax(tyroon[0:i + 1, close]) != np.amin(tyroon[0:i + 1, close]):
            entry.williamR = -100 * (np.amax(tyroon[0:i + 1, close]) - t[close]) / (
                np.amax(tyroon[0:i + 1, close]) - np.amin(tyroon[0:i + 1, close]))
        else:
            entry.williamR = 0

        # Ease of Movement
        if High != 0 and Low != 0 and t[volume] != 0:
            entry.easeMove = ((High + Low) / 2 - (previousHigh + previousLow) / 2) / ((t[volume] / 100000) / (High - Low))
        else:
            entry.easeMove = 0

        # Negative Volume Index
        if tyroon[i - 1, close] > t[close]:
            cumulativeNVI = cumulativeNVI + \
                ((tyroon[i - 1, close] - t[close]) / t[close])
        entry.negVolIndex = cumulativeNVI

        # Force Index
        entry.forceIndex = (t[close] - tyroon[i - 1, close]) * t[volume]

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

        # Comodity Channel Index:
        typical = (High + Low + t[close]) / 3
        meanDeviation = mean_abs_dev(tyroon[i - 10:i + 1, close])
        entry.commChanIndex = (
            typical - entry.day10disparity) / (0.015 * meanDeviation) if meanDeviation != 0.0 else 0.0

    if i >= N:
        # Accumulation Distribution Line:
        mFlowMultplier = ((t[close] - Low) - (High - t[close])) / (High - Low)
        mFlowVolume = mFlowMultplier * t[volume]
        previousADL = previousADL + mFlowVolume
        entry.accDistrLine = previousADL

        # Chaikin Money Flow:
        mFlowVolumeChaikin = mFlowMultplier * t[volume]
        entry.chaikinMF = mFlowVolumeChaikin / sp.sum(tyroon[i - N:i + 1, volume]) if sp.sum(tyroon[i - N:i + 1, volume]) != 0.0 else 0.0

       # Ultimate Oscillator:
        BP = tyroon[i - N:i, close] - Low
        TR = High - Low
        entry.ultimateOsc = 100 * \
            ((4 * np.mean(sp.sum(BP[0:1 / 3 * N + 1]) / TR)) + (2 * np.mean(
                sp.sum(BP[0:2 / 3 * N + 1]) / TR)) + np.mean(sp.sum(BP) / TR)) / (4 + 2 + 1)

        # Standard Deviation:
        entry.stdDev = standardDeviation

        # Slope:
        entry.slope = (t[close] - tyroon[i - N, close]) / N

        # Rate of Change:
        entry.rateChange = 100 * (t[close] - tyroon[i - N, close]) / tyroon[i - N, close]

        # Money Flow Index
        negMFlow = []
        posMFlow = []
        for j in xrange(N - 1):
            if tyroon[i - N + j, close] > tyroon[i - N + j + 1, close]:
                negMFlow.append(tyroon[i - N + j + 1])
            else:
                posMFlow.append(tyroon[i - N + j + 1])
        negMFlow = np.asarray(negMFlow)
        posMFlow = np.asarray(posMFlow)
        if len(negMFlow) > 0:
            negMFlow = sp.sum(negMFlow[:, volume]) * (
                np.average(negMFlow[:, close]) + High + Low) / 3 + 1
        else:
            negMFlow = 1
        if len(posMFlow) > 0:
            posMFlow = sp.sum(posMFlow[:, volume]) * (
                np.average(posMFlow[:, close]) + High + Low) / 3 + 1
        else:
            posMFlow = 1
        mFlowRatio = posMFlow / negMFlow
        entry.monneyFI = 100 - (100 / (1 + mFlowRatio))

        # Momentum:
        entry.momentum = (t[close] / tyroon[i - N][close]) * 100

        #Stochastik % K
        entry.stochK = 100 * (t[close] - Low) / (High - Low)

        # Parabolic SAR
        if extremePoint < np.amax(tyroon[i - N, high]):
            extremePoint = np.amax(tyroon[i - N, high])
            previousAF = previousAF + \
                0.02 if (previousAF + 0.02) <= 0.2 else 0.2
        if i == 0:
            previousSar = t[low]
        entry.paraSar = previousSar

        # Detrended Price Oscillator
        entry.detrPriceOsc = tyroon[
            i - ((N / 2) + 1), close] - entry.day10disparity

        # Average True Range:
        trueRange = max(High - Low, High - tyroon[
                        i - 1, close], Low - tyroon[i - 1, close])
        entry.avTrueRange = previousATR

        # updating the time period:
        if i % N == 0:
            previousATR = ((previousATR * 13) + trueRange) / 14
            previousSar = previousSar + \
                previousAF * (extremePoint - previousSar)
            standardDeviation = np.mean(tyroon[i - N:i + 1, close])
            standardDeviation = tyroon[i - N:i + 1, close] - standardDeviation
            standardDeviation = sp.sum(
                standardDeviation * standardDeviation) / N
            standardDeviation = standardDeviation * standardDeviation

    if i >= 2 * N:
        # Percent Volume Oscillator:
        volNEMA = volNEMA + smoothing * \
            (np.average(tyroon[i - N:i + 1, volume]) - volNEMA)
        vol2NEMA = vol2NEMA + smoothing * \
            (np.average(tyroon[i - (2 * N):i, volume]) - vol2NEMA)
        entry.percVolOsc = 100 * (volNEMA - vol2NEMA) / vol2NEMA

    entry.save()
    i = i + 1
