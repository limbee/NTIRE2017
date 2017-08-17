import os
import math

targetPath = '../flickr'
li = os.listdir(targetPath)
print('>> There are {} images'.format(len(li)))
#nDigit = int(math.ceil(math.log10(len(li)+1)))
nDigit = 6
print('  -> File name will have {} digits'.format(nDigit))
exName = '?'
for i in range(nDigit-1):
	exName = exName + '?'
print('  -> Rename as format: {}.png'.format(exName))

def newName(idx):
	name = ''
	nZeros = int(nDigit - math.ceil(math.log10(idx+1)))
	for i in range(nZeros):
		name += str(0)
	return name + str(idx) + '.png'
	
	
idx = 0
fileList = [v for v in os.listdir(targetPath)]
for filename in fileList:
	if filename.endswith('.png'):
		idx += 1
		rename = os.path.join(targetPath,newName(idx))
		filename = os.path.join(targetPath, filename)
		#print(os.path.join(targetPath, filename), rename)
		if not os.path.isfile(rename):
			os.rename(filename, rename)
		else:
			print('Warning! already {} exists in {}. Skip this file...'.format(rename, targetPath))

