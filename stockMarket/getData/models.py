#-*- coding: utf-8 -*-
from django.db import models

# Create your models here.

class Weather(models.Model):
    temperature = models.IntegerField()
    humidity = models.IntegerField()
    windSpeed = models.IntegerField()
    pressure = models.IntegerField()
    day = models.IntegerField()
    month = models.IntegerField()
    year = models.IntegerField()
    
    def __unicode__(self):
        return u'' + str(self.day) + '/' + str(self.month) + '/' + str(self.year)