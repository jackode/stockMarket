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
iter = 0
for iter in range(13):
	year = 2001 + iter
	print 'Results for:' + input + str(day) + '/' + str(month) + '/' + str(year)

	query = wae.CreateQuery(
	    input + str(day) + '/' + str(month) + '/' + str(year))
	result = wae.PerformQuery(query)
	data = wap.WolframAlphaQueryResult(result)
	jsonResult = data.JsonResult()
	data = decoder.decode(jsonResult)
	startIndex = len(data[18]) - 15

	temperature = extractIntValue(7)
	print temperature
	humidity = extractIntValue(10 + startIndex)
	print humidity
	pressure = extractIntValue(11 + startIndex)
	print pressure
	windSpeed = extractIntValue(12 + startIndex)
	print windSpeed