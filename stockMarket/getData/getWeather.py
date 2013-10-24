#-*- coding: utf-8 -*-
import json, re
from getData import wap
from getData.models import Weather


server = 'http://api.wolframalpha.com/v1/query.jsp'
appid = 'G93PW9-9H6AJERR6A'
input = 'weather Jorhat, India on'
decoder = json.JSONDecoder()

wae = wap.WolframAlphaEngine(appid, server)

#in loop
query = wae.CreateQuery(input + '23' + '/' + '10' + '/' + '2012')
result = wae.PerformQuery(query)
data = wap.WolframAlphaQueryResult(result)
jsonResult = data.JsonResult()
data = decoder.decode(jsonResult)

temperature =  data[18][7][3][2][1]
index = temperature.index('average')
temperature = temperature[index:index+15]
temperature = re.findall(r'\d+', temperature)
print temperature[0]

humidity =  data[18][11][3][2][1]
index = humidity.index('average')
humidity = humidity[index:index+15]
humidity = re.findall(r'\d+', humidity)
print humidity[0]

pressure =  data[18][12][3][2][1]
index = pressure.index('average')
pressure = pressure[index:index+15]
pressure = re.findall(r'\d+', pressure)
print pressure[0]

windSpeed =  data[18][13][3][2][1]
index = windSpeed.index('average')
windSpeed = windSpeed[index:index+15]
windSpeed = re.findall(r'\d+', windSpeed)
print windSpeed[0]
#out of loop