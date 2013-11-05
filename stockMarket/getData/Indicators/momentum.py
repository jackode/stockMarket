#-*- coding: utf-8 -*-
from getData.models import *
import scipy as sp

db = TyroonStock
tyroon = db.objects.all()


def getNDayAgo(n, day, month, year):
    if day <= n:
        day = 30 + day - n
        if month == 1:
            month = 12
            year = year - 1
        else:
            month = month - 1
    else:
        day = day - n
    try:
        return db.objects.get(day=day, month=month, year=year)
    except:
        try:
            day = day + 2
            return db.objects.get(day=day, month=month, year=year)
        except:
            return None

##################Testing only:
tyroon = tyroon[:5]
#####################


for t in reversed(tyroon):
    entry = Feature35(
        day=t.day,
        month=t.month,
        year=t.year
    )

    #Momentum:
    temp = getNDayAgo(7, t.day, t.month, t.year)
    if temp != None:
        entry.momentum = (t.close / temp.close) * 100
    #5 day disparity:
    
    entry.day5disparity = (t.close/.....) * 100

    entry.save()
