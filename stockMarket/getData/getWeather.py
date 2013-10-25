#-*- coding: utf-8 -*-
import json
import re
from getData import wap
from getData.models import Weather

def extractIntValue(position):
	if len(data[18]) < position or position < 7:
		return None
	text = data[18][position][3][2][1]
	index = text.find('average')
	if index == -1:
		return None
	else:
		text = text[index:]
		text = re.findall(r'\d+', text)
    	return text[0]


server = 'http://api.wolframalpha.com/v1/query.jsp'
appid = 'G93PW9-9H6AJERR6A'
input = 'weather Jorhat, India '
decoder = json.JSONDecoder()

wae = wap.WolframAlphaEngine(appid, server)
month = 12
day = 24
year = 2010
for year in range(2006, 2007):
	for month in range(1, 13):
		for day in range(1, 31):
			if not(month == 2 and day > 28):
				print 'Results for:' + input + str(day) + '/' + str(month) + '/' + str(year)

				query = wae.CreateQuery(
				    input + str(day) + '/' + str(month) + '/' + str(year))
				result = wae.PerformQuery(query)
				data = wap.WolframAlphaQueryResult(result)
				jsonResult = data.JsonResult()
				data = decoder.decode(jsonResult)
				startIndex = len(data[18]) - 15

				temperature = extractIntValue(7)
				humidity = extractIntValue(10 + startIndex)
				pressure = extractIntValue(11 + startIndex)
				windSpeed = extractIntValue(12 + startIndex)

				weather = Weather()
				weather.temperature = temperature
				weather.pressure = pressure
				weather.humidity = humidity
				weather.windSpeed = windSpeed
				weather.day = day
				weather.month = month
				weather.year = year
				weather.save()