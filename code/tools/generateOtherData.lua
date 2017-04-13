require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'

local cmd = torch.CmdLine()
cmd:option('-type',         'val', 	    'demo type: bench | test | val')
cmd:option('-dataset',      'DIV2K',    'external train dataset')
cmd:option('-model_dir',    'downsamplers', 'unknown downsampling model directory')
cmd:option('-degrade',      'unknown',  'degrading opertor: bicubic | unknown')
cmd:option('-scale',        2,          'scale factor: 2 | 3 | 4')
cmd:option('-scaleSwap',    -1,         'Model swap')
cmd:option('-gpuid',	    1,		    'GPU id for use')
cmd:option('-nGPU',         1,          'Number of GPUs to use by default')
cmd:option('-dataDir',	    '/var/tmp', 'data directory')

local opt = cmd:parse(arg or {})

-- imgdir
-- 

local modelname = 'donwsampler_x' .. scale .. '.t7'
local model = torch.load(paths.concat(opt.model_dir, modelname))

local save_dir = paths.concat(opt.datadir, '../X' .. opt.scale)
if not paths.dirp(save_dir) then
	paths.mkdir(save_dir)
end
for filename in paths.iterfiles(opt.datadir) do
	local imgname = paths.concat(opt.datadir, filename)
	local HR = image.load(imgname, 3, 'byte'):float():cuda()

	local LR = model:forward(HR)
	local savename = paths.concat(save_dir, filename:sub(1, -5) .. 'x' .. opt.scale)
	image.save(savename, LR)

end




