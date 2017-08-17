require 'cunn'
require 'cudnn'
require 'image'

local saveAs = '.png'
local nGPU = 1

local modelDir = '../../../downsamplers'
local apath = '/var/tmp/dataset'
local hrDir = paths.concat(apath, 'DIV2K/DIV2K_train_HR')
local lrDir = paths.concat(apath, 'DIV2K/DIV2K_train_LR_unknown_augment')

for sc = 2,4 do
	print('scale: ' .. sc)
	local modelname = 'downsampler_x' .. sc .. '.t7'
	local model = torch.load(paths.concat(modelDir, modelname)):cuda()
    if nGPU > 1 then
        local gpus = torch.range(1, nGPU):totable()
        local dpt = nn.DataParallelTable(1, true, true):add(model, gpus)
        model = dpt:cuda()
    end
    model:evaluate()

	local save_dir = paths.concat(lrDir, 'X' .. sc)
	if not paths.dirp(save_dir) then
		paths.mkdir(save_dir)
	end
    nBatch = nGPU * 4
	for filename in paths.iterfiles(hrDir) do
        local imgname = paths.concat(hrDir, filename)
        print(imgname)
        local HR = image.load(imgname, 3, 'byte'):float()
        local fileName = filename:split('%.')[1] .. 'x' .. sc
        local inputTable = {}
        for i = 1, 8 do
            if i == 1 then
                inputTable[i] = HR
            elseif i == 2 then
                inputTable[i] = image.vflip(HR)
            elseif i > 2 and i <= 4 then
                inputTable[i] = image.hflip(inputTable[i - 2])
            elseif i > 4 then
                inputTable[i] = inputTable[i - 4]:transpose(2, 3)
            end
        end
		for i = 1, 8, nBatch do
            c, h, w = unpack(inputTable[i]:size():totable())
            local inputBatch = torch.CudaTensor(nBatch, c, h, w)
            for j = 1, nBatch do
                inputBatch[j]:copy(inputTable[i + j - 1]:cuda())
            end
            local LR = model:forward(inputBatch)
            LR:add(0.5):floor():div(255)
            for j = 1, nBatch do
                local savename = paths.concat(save_dir, fileName .. '_' .. i + j - 1 .. '.png')
                image.save(savename, LR[j]:clone())
            end
            model:clearState()
            LR = nil
            collectgarbage()
            collectgarbage()
        end
	end
end
