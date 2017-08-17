import os
import threading
import urllib2
import cStringIO
from PIL import Image
from scipy.misc import imsave, imresize
import numpy as np
from socket import timeout
from flickrapi import FlickrAPI

FLICKR_PUBLIC = '' # use your own key
FLICKR_SECRET = '' # use your own key

querySet = []
querySet.append(['animal','plants','city','nature'])
querySet.append([
'fish','mammal','bird','insect',
'fruits','flower','tree',
'street','cityscape','building','market',
'seaside','lake','valley','forest',
'food','crowd'])
querySet.append([
'mushroom','coral','rose','orange','onion','leaf',
'lion','tiger','horse','cat','monkey','peacock','fox','bear','squirrel','bee','starfish','spider',
'apartment','architecture','train','car','university','garden','park',
'coast','beach','climbing','mountain','garden','snow','bread',
'photographic','hawai','dubai','vegas','seoul'])
querySet.append([
'jungle','landscape','spring','summer','autumn','winter','view','sightseeing','scenery',
'hotel','bar','library','photographs','audience','interior','army','tribes','junk','trash',
'pabble','centralpark','machinery','semiconductor','zoo','farm','hockey','tourdefrance','festival',
'museum','poverty','war','microscopic','structure','orchestra','traditional'])
querySet.append([
'school','blossom','cherry','Reykjavik','milkyway','restaurant','penguin','rabbit','italy','cow',
'wallpaper','strawberry','boar','temple','sculpture','beauty','complicated','bush','themepark','parrot',
'tourist','swiss','korea','india','tokyo','assam','dog','kebob','salad','fruitstore',
'wheat','newyork','corn','kingcrab','graphity','timber','seashore','cremona','playground','queenstown'])

nPage = [4,3,2,2,2]
per_page= 500

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
extras = 'url_o'

def byteify(input):
	if isinstance(input, dict):
		return {byteify(key): byteify(value) for key, value in input.iteritems()}
	elif isinstance(input, list):
		return [byteify(element) for element in input]
	elif isinstance(input, unicode):
		return input.encode('utf-8')
	else:
		return input

baseDir = '../flickr'
if not os.path.isdir(baseDir):
	os.mkdir(baseDir)

def getter(res, query, i):
	if not ('height_o' in res[i] and 'width_o' in res[i] and 'url_o' in res[i]):
		print('\t-> Skip this image: cannot access the image')
		return
	'''
	if not '.png' in res[i]['url_o']:
		print('\t-> Skip this image: image format is ' + res[i]['url_o'].split('.')[-1])
		return
	'''

	name = res[i]['title']
	height = int(res[i]['height_o'])
	width = int(res[i]['width_o'])

	print('[{}/{}][{}/{}](query: {})[{}/{}]: {}'.format(
		iQuerySet + 1, len(querySet),
		iQuery, len(queries), query,
		(page - 1) * per_page + i + 1, len(res) * nPage[iQuerySet], name[1:30]))
	if max(height, width) < 1530:
		print('\t-> Skip this image: size is too small (height={}, width={})'.format(height, width))
		return
	elif float(height)/float(width) < 0.5 or float(height)/float(width) > 2:
		print('\t-> Too large aspect ratio')
		return

	try:
		url = res[i]['url_o']
		imgFile = cStringIO.StringIO(urllib2.urlopen(url).read())
		img = Image.open(imgFile)
		img = np.asarray(img)
		if img.shape[2] != 3:
			return
		
		if max(height, width) > 2040:
			scaleDown = 2040.0 / max(height, width)
			print '\t-> Resize *%.2f: ' % scaleDown, '({}, {}) -> ({}, {})'.format(
					height, width, int(height * scaleDown), int(width * scaleDown))
			img = imresize(img, float(scaleDown), interp='bicubic')
		
		tH = 12 * int(height / 12)
		tW = 12 * int(width / 12)
		img = img[:tH, :tW, :]
		imsave(os.path.join(baseDir, query + '_' + str((page-1) * per_page + i) + '.png'), img)
	except:
		print('\t-> Error: loading image')

iQuerySet = 0
for queries in querySet:

	iQuery = 1
	for query in queries:
		'''
		queryDir = os.path.join(baseDir, query)
		if not os.path.isdir(queryDir):
			os.mkdir(queryDir)
		'''

		for page in range(1, nPage[iQuerySet] + 1):
			try:
				res = flickr.photos.search(text=query, per_page=per_page, page=page, content_type=1, extras=extras)
			except:
				continue

			res = byteify(res)['photos']['photo']

			threads = []
			for i in range(len(res)):
				t = threading.Thread(target=getter, args=(res, query, i))
				t.start()
				threads.append(t)

			for t in threads:
				t.join()
		iQuery = iQuery + 1

		print('\n\nCompleted ' + query + '!!\n\n')
	
	iQuerySet += 1
