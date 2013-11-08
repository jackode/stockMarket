#-*- coding: utf-8 -*-
from getData.models import *
import scipy as sp
import numpy as np
import math

weather = sp.genfromtxt('../CSV Data/dbDump/dweather_jorhat.csv', delimiter=';')
id = 0
temp = 1
hum = 2
wind = 3
press = 4
day = 5
month = 6
year = 7

previous = 0
for row in weather:
    entry = ddWeather(
        day=row[day],
        month=row[month],
        year=row[year]
    )
    if not isinstance(previous, int):
        result = row - previous
        entry.temperature = result[temp] if not math.isnan(result[temp]) else None
        entry.humidity = result[hum] if not math.isnan(result[hum]) else None
        entry.windSpeed = result[wind] if not math.isnan(result[wind]) else None
        entry.pressure = result[press] if not math.isnan(result[press]) else None
    else:
        entry.temperature = 0
        entry.humidity = 0
        entry.windSpeed = 0
        entry.pressure = 0

    entry.save()
    previous = row


# USE THIS FOR FEATURES:
# weather = sp.genfromtxt('../CSV Data/dbDump/features_tyroon.csv', delimiter=';')
# id = 0
# day = 1
# month = 2
# year = 3
# momentum = 4
# day5disparity = 5
# day10disparity = 6
# stochK = 7
# priceVolumeTrend = 8
# movAverageExp = 9
# paraSar = 10
# accDistrLine = 11
# avTrueRange = 12
# indicB = 13
# commChanIndex = 14
# chaikinMF = 15
# detrPriceOsc = 16
# easeMove = 17
# forceIndex = 18
# macd = 19
# monneyFI = 20
# negVolIndex = 21
# percVolOsc = 22
# priceRelWarrent = 23
# priceRelAsian = 24
# priceRelDiana = 25
# priceRelTenren = 26
# rateChange = 27
# relStrengthI = 28
# slope = 29
# stdDev = 30
# stochOsc = 31
# stochRSI = 32
# ultimateOsc = 33
# williamR = 34

# previous = 0
# for row in weather:
#     entry = dFeature35(
#         day=row[day],
#         month=row[month],
#         year=row[year]
#     )
#     if not isinstance(previous, int):
#         result = row - previous
#         entry.momentum = result[momentum] if not math.isnan(result[momentum]) else None
#         entry.day5disparity = result[day5disparity] if not math.isnan(result[day5disparity]) else None
#         entry.day10disparity = result[day10disparity] if not math.isnan(result[day10disparity]) else None
#         entry.stochK = result[stochK] if not math.isnan(result[stochK]) else None
#         entry.priceVolumeTrend = result[priceVolumeTrend] if not math.isnan(result[priceVolumeTrend]) else None
#         entry.movAverageExp = result[movAverageExp] if not math.isnan(result[movAverageExp]) else None
#         entry.paraSar = result[paraSar] if not math.isnan(result[paraSar]) else None
#         entry.accDistrLine = result[accDistrLine] if not math.isnan(result[accDistrLine]) else None
#         entry.avTrueRange = result[avTrueRange] if not math.isnan(result[avTrueRange]) else None
#         entry.indicB = result[indicB] if not math.isnan(result[indicB]) else None
#         entry.commChanIndex = result[commChanIndex] if not math.isnan(result[commChanIndex]) else None
#         entry.chaikinMF = result[chaikinMF] if not math.isnan(result[chaikinMF]) else None
#         entry.detrPriceOsc = result[detrPriceOsc] if not math.isnan(result[detrPriceOsc]) else None
#         entry.easeMove = result[easeMove] if not math.isnan(result[easeMove]) else None
#         entry.forceIndex = result[forceIndex] if not math.isnan(result[forceIndex]) else None
#         entry.macd = result[macd] if not math.isnan(result[macd]) else None
#         entry.monneyFI = result[monneyFI] if not math.isnan(result[monneyFI]) else None
#         entry.negVolIndex = result[negVolIndex] if not math.isnan(result[negVolIndex]) else None
#         entry.percVolOsc = result[percVolOsc] if not math.isnan(result[percVolOsc]) else None
#         entry.priceRelWarrent = result[priceRelWarrent] if not math.isnan(result[priceRelWarrent]) else None
#         entry.priceRelAsian = result[priceRelAsian] if not math.isnan(result[priceRelAsian]) else None
#         entry.priceRelDiana = result[priceRelDiana] if not math.isnan(result[priceRelDiana]) else None
#         entry.priceRelTenren = result[priceRelTenren] if not math.isnan(result[priceRelTenren]) else None
#         entry.rateChange = result[rateChange] if not math.isnan(result[rateChange]) else None
#         entry.relStrengthI = result[relStrengthI] if not math.isnan(result[relStrengthI]) else None
#         entry.slope = result[slope] if not math.isnan(result[slope]) else None
#         entry.stdDev = result[stdDev] if not math.isnan(result[stdDev]) else None
#         entry.stochOsc = result[stochOsc] if not math.isnan(result[stochOsc]) else None
#         entry.stochRSI = result[stochRSI] if not math.isnan(result[stochRSI]) else None
#         entry.ultimateOsc = result[ultimateOsc] if not math.isnan(result[ultimateOsc]) else None
#         entry.williamR = result[williamR] if not math.isnan(result[williamR]) else None

#     else:
#         entry.momentum = None
#         entry.day5disparity = None
#         entry.day10disparity = None
#         entry.stochK = None
#         entry.priceVolumeTrend = None
#         entry.movAverageExp = None
#         entry.paraSar = None
#         entry.accDistrLine = None
#         entry.avTrueRange = None
#         entry.indicB = None
#         entry.commChanIndex = None
#         entry.chaikinMF = None
#         entry.detrPriceOsc = None
#         entry.easeMove = None
#         entry.forceIndex = None
#         entry.macd = None
#         entry.monneyFI = None
#         entry.negVolIndex = None
#         entry.percVolOsc = None
#         entry.priceRelWarrent = None
#         entry.priceRelAsian = None
#         entry.priceRelDiana = None
#         entry.priceRelTenren = None
#         entry.rateChange = None
#         entry.relStrengthI = None
#         entry.slope = None
#         entry.stdDev = None
#         entry.stochOsc = None
#         entry.stochRSI = None
#         entry.ultimateOsc = None
#         entry.williamR = None
#     entry.save()
#     previous = row
