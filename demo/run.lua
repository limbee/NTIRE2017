require 'cunn'
require 'cudnn'
require 'image'


local cmd = torch.CmdLine()
cmd:option('-type',		'bench',	'demo type: bench | test')
cmd:option('-model',	'resnet',	'model type: resnet | vdsr')
cmd:option('-gpuid',	2,			'GPU id for use')
local opt = cmd:parse(arg or {})
local inputSize = opt.model=='resnet' and 'small' or 'big'
local now = os.date('%Y-%m-%d_%H-%M-%S')
cutorch.setDevice(opt.gpuid)

for modelFile in paths.iterfiles('model') do
	if modelFile:sub(1,1)=='.' then goto continue end
	
	local model = torch.load(paths.concat('model',modelFile)):cuda()
	model:evaluate()

	local modelName = modelFile:split('%.')[1]
	print('>> test on ' .. modelName .. ' ......')
	paths.mkdir(paths.concat('img_output',modelName))

	if opt.type=='bench' then
		local dir = paths.concat('../dataset/benchmark/',inputSize)
		for subDir in paths.iterdirs(dir) do
			local inpDir = paths.concat(dir,subDir)
			local resDir = paths.concat('img_output',modelName,subDir)
			local tarDir = paths.concat('img_target',modelName,subDir)
			local gtDir = paths.concat('../dataset/benchmark',subDir)
			paths.mkdir(resDir)
			paths.mkdir(tarDir)

			for imgFile in paths.iterfiles(inpDir) do
				print('\t' .. paths.concat(inpDir,imgFile))
				local input = image.load(paths.concat(inpDir,imgFile)):mul(255)
				local gt = pcall(function() image.load(paths.concat(gtDir,imgFile)) end)
					and image.load(paths.concat(gtDir,imgFile))
					or image.load(paths.concat(gtDir,imgFile:split('%.')[1]..'.jpg'))

				if input:dim()==2 or (input:dim()==3 and input:size(1)==1) then input = input:repeatTensor(3,1,1) end
				if gt:dim()==2 or (gt:dim()==3 and gt:size(1)==1) then gt = gt:repeatTensor(3,1,1) end
				input = input:view(1,table.unpack(input:size():totable()))

				-- This function prevents the gpu memory from overflowing
				-- by passing the input layer-by-layer through the network.
				local function getOutput(input,model)
					local output
					if model.__typename:find('Concat') then
						output = {}
						for i=1,model:size() do
							table.insert(output,getOutput(input,model:get(i)))
						end
					elseif model.__typename:find('Sequential') then
						output = input
						for i=1,#model do
							output = getOutput(output,model:get(i))
						end
					else
						output = model:forward(input)
					end
					return output
				end
				local output = getOutput(input:cuda(),model):squeeze():div(255)

				gt = gt[{{},{1,output:size(2)},{1,output:size(3)}}]
				image.save(paths.concat(resDir,imgFile),output)
				image.save(paths.concat(tarDir,imgFile:split('%.')[1] .. '.png'),gt)
				collectgarbage();
			end
		end
	else
	end

	::continue::
end