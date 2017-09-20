from PIL import Image
import numpy as np
import skimage.io as sio
import imageio
import os

nDiv = 20
width = 5
scale = 4
img_list = [f for f in os.listdir('img_input') if (os.path.isfile('img_input/' + f) and f.endswith('.png'))]

for img_name in img_list:
	print(img_name)
	img_inp = Image.open('img_input/' + img_name)
	img_tar = Image.open('img_output/EDSR_x{}/myImages/X{}/'.format(scale, scale) + img_name)

	(w, h) = img_inp.size
	ww, hh = w * scale, h * scale
	img_inp = img_inp.resize((ww, hh), Image.NEAREST)

	img_inp = np.asarray(img_inp)
	img_tar = np.asarray(img_tar)

	output = list(range(2 * (nDiv - 1)))
	for i in range(nDiv - 1):
		frame = np.array(img_inp)
		idx = int(img_inp.shape[1] * (i + 1) / nDiv)
		frame[:, :idx - width, :] = img_inp[:, :idx - width, :]
		frame[:, idx + width:, :] = img_tar[:, idx + width:, :]
		frame[:, idx-width:idx+width, :] = np.array([[[0,0,255]]])
		output[i] = frame
		output[2 * (nDiv - 1) - i - 1] = frame
	
	(name, ext) = os.path.splitext(img_name)
	imageio.mimwrite('gif/{}.gif'.format(name), output, fps=nDiv, palettesize=256)

