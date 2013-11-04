#-*- coding: utf-8 -*-
from django.db import models

# Create your models here.


class Weather(models.Model):
    temperature = models.SmallIntegerField(null=True, blank=True)
    humidity = models.SmallIntegerField(null=True, blank=True)
    windSpeed = models.SmallIntegerField(null=True, blank=True)
    pressure = models.SmallIntegerField(null=True, blank=True)
    day = models.SmallIntegerField()
    month = models.SmallIntegerField()
    year = models.SmallIntegerField()

    def __unicode__(self):
        return u'' + str(self.day) + '/' + str(self.month) + '/' + str(self.year)


class Stock(models.Model):
    day = models.SmallIntegerField()
    month = models.SmallIntegerField()
    year = models.SmallIntegerField()
    value = models.DecimalField(null=True, blank=True, max_digits=7, decimal_places=4)
    volume = models.DecimalField(null=True, blank=True, max_digits=13, decimal_places=4)

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


class dWeather(Weather):
    pass


class ddWeather(Weather):
    pass

class Feature35(models.Model):
    pass