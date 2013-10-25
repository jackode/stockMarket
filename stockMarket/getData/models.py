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
    day = models.SmallIntegerField(null=True, blank=True)
    month = models.SmallIntegerField(null=True, blank=True)
    year = models.SmallIntegerField(null=True, blank=True)
    value = models.SmallIntegerField(null=True, blank=True)

    def __unicode__(self):
        return u'' + str(self.day) + '/' + str(self.month) + '/' + str(self.year)