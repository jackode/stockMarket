#-*- coding: utf-8 -*-
from django.db import models

# Create your models here.


class Feature(models.Model):
    day = models.SmallIntegerField()
    month = models.SmallIntegerField()
    year = models.SmallIntegerField()
    momentum = models.FloatField(
        null=True, blank=True)
    day5disparity = models.FloatField(
        null=True, blank=True)
    day10disparity = models.FloatField(
        null=True, blank=True)
    stochK = models.FloatField(
        null=True, blank=True)
    priceVolumeTrend = models.FloatField(
        null=True, blank=True)
    movAverageExp = models.FloatField(
        null=True, blank=True)
    paraSar = models.FloatField(
        null=True, blank=True)
    accDistrLine = models.FloatField(
        null=True, blank=True)
    avTrueRange = models.FloatField(
        null=True, blank=True)
    indicB = models.FloatField(
        null=True, blank=True)
    commChanIndex = models.FloatField(
        null=True, blank=True)
    chaikinMF = models.FloatField(
        null=True, blank=True)
    detrPriceOsc = models.FloatField(
        null=True, blank=True)
    easeMove = models.FloatField(
        null=True, blank=True)
    forceIndex = models.FloatField(
        null=True, blank=True)
    macd = models.FloatField(
        null=True, blank=True)
    monneyFI = models.FloatField(
        null=True, blank=True)
    negVolIndex = models.FloatField(
        null=True, blank=True)
    percVolOsc = models.FloatField(
        null=True, blank=True)
    priceRelWarrent = models.FloatField(
        null=True, blank=True)
    priceRelAsian = models.FloatField(
        null=True, blank=True)
    priceRelDiana = models.FloatField(
        null=True, blank=True)
    priceRelTenren = models.FloatField(
        null=True, blank=True)
    rateChange = models.FloatField(
        null=True, blank=True)
    relStrengthI = models.FloatField(
        null=True, blank=True)
    slope = models.FloatField(
        null=True, blank=True)
    stdDev = models.FloatField(
        null=True, blank=True)
    stochOsc = models.FloatField(
        null=True, blank=True)
    stochRSI = models.FloatField(
        null=True, blank=True)
    ultimateOsc = models.FloatField(
        null=True, blank=True)
    williamR = models.FloatField(
        null=True, blank=True)

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
