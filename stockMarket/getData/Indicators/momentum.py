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
extremePoint = previousSar = previousAF = previousADL = 0.0

# Testing only:
# Also, CHANGE DIANA WITH TYROON !
tyroon = tyroon[:25]
#

tyroon = tyroon[::-1]
for t in tyroon:
    entry = Feature35(
        day=t[day],
        month=t[month],
        year=t[year]
    )

    # Exponential Moving Average:
    smoothing = 2.0 / 11.0
    previousEMA = ((t[close] - previousEMA) * smoothing) + previousEMA
    entry.movAverageExp = previousEMA

    if i >= 1:
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

    if i >= N:
        Low = np.amin(tyroon[i - N:i, close])
        High = np.amax(tyroon[i - N:i, close])

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
        else:
            previousSar = previousSar + \
                previousAF * (extremePoint - previousSar)
        entry.paraSar = previousSar

        # Accumulation Distribution Line:
        mFlowMultplier = ((t[close] - Low) - (High - t[close])) / (High - Low)
        mFlowVolume = mFlowMultplier * sp.sum(tyroon[i -N:i, volume])
        previousADL = previousADL + mFlowVolume
        entry.accDistrLine = previousADL


    entry.save()
    i = i + 1
