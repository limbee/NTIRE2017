require 'nn'
require 'cunn'
require 'cudnn'
require 'image'


local cmd = torch.CmdLine()
cmd:option('-type',	'test',	        'demo type: bench | test')
cmd:option('-model',    'resnet',	'model type: resnet | vdsr')
cmd:option('-degrade',  'bicubic',      'degrading opertor: bicubic | unknown')
cmd:option('-scale',    2,              'scale factor: 2 | 3 | 4')
cmd:option('-gpuid',	2,		'GPU id for use')
local opt = cmd:parse(arg or {})
local now = os.date('%Y-%m-%d_%H-%M-%S')
cutorch.setDevice(opt.gpuid)

local testList = {}

for modelFile in paths.iterfiles('model') do
    if modelFile:sub(1,1 ) ~= '.' then
        local subDir = ''
        if (opt.type == 'bench') then
	    
        elseif (opt.type == 'test') then
            --for DIV2K dataset
            dataDir = '/var/tmp/dataset/DIV2K/DIV2K_valid_LR_' .. opt.degrade  .. '/X' .. opt.scale
            if (opt.model == 'vdsr') then
                dataDir = dataDir .. 'b'
            end
            for testFile in paths.iterfiles(dataDir) do
                if (string.find(testFile, '.png')) then
                    table.insert(testList, {dataDir .. '/' .. testFile, testFile})
                end
            end
        end

        local model = torch.load(paths.concat('model',modelFile)):cuda()
        local modelName = modelFile:split('%.')[1]
	print('>> test on ' .. modelName .. ' ......')
	model:evaluate()
        local timer = torch.Timer()
        
        for i = 1, #testList do
            local input = image.load(testList[i][1]):mul(255)
            --local target = image.load(testList[i][2]):mul(255)
            if (input:dim() == 2 or (input:dim() == 3 and input:size(1) == 1)) then
                input:repeatTensor(input, 3, 1, 1)
            end
            --if (target:dim() == 2 or (target:dim() == 3 and target:size(1) == 1)) then
            --    target:repeatTensor(target, 3, 1, 1)
            --end
            input:view(input, 1, table.unpack(input:size():totable()))

            local __model = model
            local function getOutput(input, model)
                local output
                if model.__typename:find('Concat') then
                    output = {}
                    for i = 1, model:size() do
                        table.insert(output, getOutput(input, model:get(i)))
                    end
                elseif model.__typename:find('Sequential') then
                    output = input
                    for i = 1, #model do
                        output = getOutput(output, model:get(i))
                    end
                else
                    output = model:forward(input):clone()
                    model = nil
                    __model:clearState()
                    collectgarbage()
                end
                return output
            end
            local output = getOutput(input:cuda(), model):squeeze(1):div(255)
            --target = target[{{}, {1, output:size(2)}, {1, output:size(3)}}]
            image.save('img_output/' .. testList[i][2], output)
            --image.save('img_target/' .. testLitt[i][2], target)
        end
        print('Elapsed time: ' .. timer:time.real())
    end
end
