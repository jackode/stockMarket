#-*- coding: utf-8 -*-
import numpy as np
from getData.models import *

db = TenRenStock
file = '../CSV Data/raw/TenRen.csv'
day, close, volume = 0, 4, 5
open, low, high, adj = 1, 2, 3, 6

data = np.genfromtxt(file, delimiter=';', dtype=None, names=True)

for row in data:
    date = row[day].split('-')
    entry = db()
    entry.day = float(date[2])
    entry.month = float(date[1])
    entry.year = float(date[0])
    entry.volume = float(row[volume])
    entry.close = float(row[close])
    entry.open = float(row[open])
    entry.low = float(row[low])
    entry.high = float(row[high])
    entry.adj = float(row[adj])
    entry.save()
