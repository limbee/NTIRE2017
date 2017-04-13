require 'cunn'
require 'cudnn'
require 'image'

local modelDir = '/home/limbee/downsamplers'
local apath = '/dataset/Flickr2K/'
local hrDir = paths.concat(apath, 'Flickr2K_HR')
local lrDir = paths.concat(apath, 'Flickr2K_LR_unknown')

for sc = 2,4 do
	print('scale: ' .. sc)
	local modelname = 'downsampler_x' .. sc .. '.t7'
	local model = torch.load(paths.concat(modelDir, modelname)):cuda()

	local save_dir = paths.concat(lrDir, 'X' .. sc)
	if not paths.dirp(save_dir) then
		paths.mkdir(save_dir)
	end

	for filename in paths.iterfiles(hrDir) do
		local imgname = paths.concat(hrDir, filename)
		print(imgname)
		local HR = image.load(imgname, 3, 'byte'):float():cuda()

		local LR = model:forward(HR)
		local savename = paths.concat(save_dir, filename:split('%.')[1] .. 'x' .. sc .. '.png')
		image.save(savename, LR:byte())

		model:clearState()
		collectgarbage()
		collectgarbage()
		collectgarbage()
	end
end
