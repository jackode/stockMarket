#-*- coding: utf-8 -*-
import numpy as np
from getData.models import *

db = IndianStock
file = '../CSV Data/raw/AsianTNE.csv'
day, close, volume = 0, 4, 5

data = np.genfromtxt(file, delimiter=';', dtype=None, names=True)

for row in data:
    date = row[day].split('-')
    entry = db()
    entry.day = float(date[2])
    entry.month = float(date[1])
    entry.year = float(date[0])
    entry.volume = float(row[volume])
    entry.value = float(row[close])
    entry.save()
