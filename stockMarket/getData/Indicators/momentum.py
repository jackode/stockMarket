#-*- coding: utf-8 -*-
from getData.models import *
import scipy as sp
import numpy as np

# db = TyroonStock
# tyroon = db.objects.all()
tyroon = sp.genfromtxt('../CSV Data/dbDump/stock_tyroon.csv', delimiter=';')
N = 14
day, month, year, open, close, low, high, adj, volume = 1, 2, 3, 4, 5, 6, 7, 8, 9
i = 0
previousEMA = 0.0
extremePoint = previousSar = previousAF = 0.0


def getNDayAgo(n, day=0, month=0, year=0):
    return dayClose[-n]
    # if day <= n:
    #     day = 30 + day - n
    #     if month == 1:
    #         month = 12
    #         year = year - 1
    #     else:
    #         month = month - 1
    # else:
    #     day = day - n
    # try:
    #     return db.objects.get(day=day, month=month, year=year)
    # except:
    #     try:
    #         day = day + 2
    #         return db.objects.get(day=day, month=month, year=year)
    #     except:
    #         return None

# Testing only:
tyroon = tyroon[:25]
#

tyroon = tyroon[::-1]
for t in tyroon:
    entry = Feature35(
        day=t[day],
        month=t[month],
        year=t[year]
    )

    # Momentum:
    if i >= N:
        entry.momentum = (t[close] / tyroon[i - N][close]) * 100

    # 5 day disparity:
    if i >= 5:
        temp = np.mean(tyroon[i - 5:i, close])
        entry.day5disparity = (t[close] / temp) * 100

    # 10 day disparity:
    if i >= 10:
        temp = np.mean(tyroon[i - 10:i, close])
        entry.day10disparity = (t[close] / temp) * 100

    #Stochastik % K
    if i >= N:
        B = np.amin(tyroon[i - N:i, close])
        H = np.amax(tyroon[i - N:i, close])
        entry.stochK = 100 * (t[close] - B) / (H - B)

    # Price Volume Trend:
    if i >= 1:
        entry.priceVolumeTrend = t[
            volume] * (t[close] - tyroon[i - 1][close]) / tyroon[i - 1][close]

    # Exponential Moving Average:
    smoothing = 2.0 / 11.0
    previousEMA = ((t[close] - previousEMA) * smoothing) + previousEMA
    entry.movAverageExp = previousEMA

    # Parabolic SAR
    if i > N: 
        if extremePoint < np.amax(tyroon[i-N, high]):
            extremePoint = np.amax(tyroon[i-N, high])
            previousAF = previousAF + 0.02 if (previousAF + 0.02) <= 0.2 else 0.2
        if i == 0:
            previousSar = t[low]
        else:
            previousSar = previousSar + previousAF*(extremePoint - previousSar)
        entry.paraSar = previousSar

    #Accumulation Distribution Line:
    
    # entry.accDistrLine = 

    entry.save()
    i = i + 1
