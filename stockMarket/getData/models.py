#-*- coding: utf-8 -*-
from django.db import models

# Create your models here.


class Feature(models.Model):
    day = models.SmallIntegerField()
    month = models.SmallIntegerField()
    year = models.SmallIntegerField()
    momentum = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    day5disparity = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    day10disparity = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    stochK = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    priceVolumeTrend = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    movAverageExp = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    paraSar = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    accDistrLine = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    avTrueRange = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    indicB = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    commChanIndex = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    chaikinMF = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    detrPriceOsc = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    easeMove = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    forceIndex = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    macd = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    monneyFI = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    negVolIndex = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    percVolOsc = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    priceRelWarrent = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    priceRelAsian = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    priceRelDiana = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    priceRelTenren = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    rateChange = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    relStrengthI = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    slope = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    stdDev = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    stochOsc = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    stochRSI = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    trueStrengthI = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    ultimateOsc = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    williamR = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)
    volIndex = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)

    def __unicode__(self):
        return u'' + str(self.day) + '/' + str(self.month) + '/' + str(self.year)

    class Meta:
        abstract = True


class W(models.Model):
    temperature = models.SmallIntegerField(null=True, blank=True)
    humidity = models.SmallIntegerField(null=True, blank=True)
    windSpeed = models.SmallIntegerField(null=True, blank=True)
    pressure = models.SmallIntegerField(null=True, blank=True)
    day = models.SmallIntegerField()
    month = models.SmallIntegerField()
    year = models.SmallIntegerField()

    def __unicode__(self):
        return u'' + str(self.day) + '/' + str(self.month) + '/' + str(self.year)

    class Meta:
        abstract = True


class Stock(models.Model):
    day = models.SmallIntegerField()
    month = models.SmallIntegerField()
    year = models.SmallIntegerField()
    open = models.DecimalField(
        null=True, blank=True, max_digits=7, decimal_places=4)
    close = models.DecimalField(
        null=True, blank=True, max_digits=7, decimal_places=4)
    low = models.DecimalField(
        null=True, blank=True, max_digits=7, decimal_places=4)
    high = models.DecimalField(
        null=True, blank=True, max_digits=7, decimal_places=4)
    adj = models.DecimalField(
        null=True, blank=True, max_digits=7, decimal_places=4)
    volume = models.DecimalField(
        null=True, blank=True, max_digits=13, decimal_places=4)

    class Meta:
        abstract = True

    def __unicode__(self):
        return u'' + str(self.day) + '/' + str(self.month) + '/' + str(self.year)


class TyroonStock(Stock):
    pass


class WarrentStock(Stock):
    pass


class IndianStock(Stock):
    pass


class TenRenStock(Stock):
    pass


class DianaStock(Stock):
    pass


class Weather(W):
    pass


class dWeather(W):
    pass


class ddWeather(W):
    pass


class Feature35(Feature):
    pass


class dFeature35(Feature):
    pass
