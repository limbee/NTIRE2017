import os
import urllib
from socket import timeout
from apiclient.discovery import build

service = build(
	'customsearch', 
	'v1', 
	developerKey='' # use your own key
)

	
nSearchPerQuery = 10
nResultPerSearch = 10
#queries = ['nature', 'jungle', 'park', 'landscape', 'forest', 'spring', 'summer', 'autumn', 'winter', 'view', 'sightseeing', 'scenery', 'architecture', 'city', 'hotel', 'bar', 'library', 'animal', 'plants', 'insect', 'photographs', 'market', 'crowd', 'audience', 'interior', 'army', 'tribes', 'food', 'junk', 'trash', 'beach', 'pabble', 'centralpark', 'machinery', 'semiconductor', 'zoo', 'farm', 'hockey', 'tourdefrance', 'festival', 'garden', 'university', 'museum', 'seoul', 'poverty', 'war', 'microscopic', 'structure', 'orchestra', 'traditional']
queries = ['park', 'landscape', 'forest', 'spring', 'summer', 'autumn', 'winter', 'view', 'sightseeing', 'scenery', 'architecture', 'city', 'hotel', 'bar', 'library', 'animal', 'plants', 'insect', 'photographs', 'market', 'crowd', 'audience', 'interior', 'army', 'tribes', 'food', 'junk', 'trash', 'beach', 'pabble', 'centralpark', 'machinery', 'semiconductor', 'zoo', 'farm', 'hockey', 'tourdefrance', 'festival', 'garden', 'university', 'museum', 'seoul', 'poverty', 'war', 'microscopic', 'structure', 'orchestra', 'traditional']

iQuery = 1
for query in queries:
	if not os.path.isdir(query):
		os.mkdir(query)

	iRes = 1
	for i in range(nSearchPerQuery):
		res_huge = service.cse().list(
			q=query,
			cx='', # use your own key
			searchType='image',
			num=nResultPerSearch,
			imgType='photo',
			#imgSize='xxlarge',
			imgSize='huge',
			imgColorType='color',
			fileType='png',
			#lowRange=11,
			#highRange=20,
			start = 10 * i + 1
		).execute()

		res_xxlarge = service.cse().list(
			q=query,
			cx='', # use your own key
			searchType='image',
			num=nResultPerSearch,
			imgType='photo',
			imgSize='xxlarge',
			#imgSize='huge',
			imgColorType='color',
			fileType='png',
			#lowRange=11,
			#highRange=20,
			start = 10 * i + 1
		).execute()

		for item in res_huge['items']:
			print('[{}/{}]({}) [1/2](huge) [{}/{}]: {}'.format(iQuery, len(queries), query, iRes, 2 * nSearchPerQuery * nResultPerSearch, item['title'].encode('utf-8')))
			try:
				urllib.urlretrieve(item['link'], query + '/' + str(iRes) + '.png')
			except:
				print('error: ' + item['link'])

			iRes += 1

		for item in res_xxlarge['items']:
			print('[{}/{}]({}) [2/2](xxlarge) [{}/{}]: {}'.format(iQuery, len(queries), query, iRes, 2 * nSearchPerQuery * nResultPerSearch, item['title'].encode('utf-8')))
			try:
				urllib.urlretrieve(item['link'], query + '/' + str(iRes) + '.png')
			except:
				print('error: ' + item['link'])

			iRes += 1

	iQuery += 1
