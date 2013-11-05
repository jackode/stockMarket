#-*- coding: utf-8 -*-
import matplotlib.pyplot as uniquePyPlot


def plot(x, y, title='', xlabel='', ylabel=''):
    uniquePyPlot.clf()
    figure = uniquePyPlot
    figure.scatter(x, y)
    figure.title(title)
    figure.xlabel(xlabel)
    figure.ylabel(ylabel)
    figure.autoscale(tight=True)
    figure.grid()
    figure.savefig(title + '.svg', format='svg')