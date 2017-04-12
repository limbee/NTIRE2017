require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require '../code/model/common'

local cmd = torch.CmdLine()
cmd:option('-type',         'val', 	    'demo type: bench | test | val')
cmd:option('-dataset',      'DIV2K',    'test dataset')
cmd:option('-mulImg',       255,        'multiply constant to input image')
cmd:option('-progress',     'true',     'show current progress')
cmd:option('-model',        'resnet',   'substring of model name')
cmd:option('-degrade',      'bicubic',  'degrading opertor: bicubic | unknown')
cmd:option('-scale',        2,          'scale factor: 2 | 3 | 4')
cmd:option('-scaleSwap',    -1,         'Model swap')
cmd:option('-gpuid',	    1,		    'GPU id for use')
cmd:option('-dataDir',	    '/var/tmp', 'data directory')
cmd:option('-selfEnsemble', 'false',    'enables self ensemble with flip and rotation')
cmd:option('-chopShave',    10,         'Shave width for chopForward')
cmd:option('-chopSize',     16e4,       'Minimum chop size for chopForward')

local opt = cmd:parse(arg or {})
opt.progress = (opt.progress == 'true')
opt.model = opt.model:split('+')
opt.selfEnsemble = (opt.selfEnsemble == 'true')

local util = require '../code/utils'(opt)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid)

--Prepare the dataset for demo
print('Preparing dataset...')
local testList = {}
local dataDir = ''
local Xs = 'X' .. opt.scale

if opt.type == 'bench' or opt.type == 'val' then
    dataDir = paths.concat(opt.dataDir, 'dataset/benchmark/small')
    for benchFolder in paths.iterdirs(dataDir) do
        if opt.type == 'bench' or (opt.type == 'val' and benchFolder == 'val') then
            local inputFolder = paths.concat(dataDir, benchFolder, X2)
            for benchFile in paths.iterfiles(inputFolder) do
                if benchFile:find('.png') then
                    table.insert(testList, 
                    {
                        setName = benchFolder,
                        from = inputFolder,
                        fileName = benchFile
                    })
                end
            end
        end
    end
elseif opt.type == 'test' then
    if opt.dataset == 'DIV2K' then
        dataDir = paths.concat(opt.dataDir, 'dataset/DIV2K/DIV2K_test_LR_' .. opt.degrade, Xs)
        for testFile in paths.iterfiles(dataDir) do
            if testFile:find('.png') then
                table.insert(testList,
                {
                    setName = 'test',
                    from = dataDir,
                    fileName = testFile
                })
            end
        end
    --You can test with your own images.
    --Please put your images in the img_input folder
    else
        for testFile in paths.iterfiles('img_input') do
            if testFile:find('.png') or testFile:find('.jp') then
                table.insert(testList,
                {
                    setName = 'myImages',
                    from = 'img_input',
                    fileName = testFile
                })
            end
        end
    end
end
table.sort(testList, function(a, b) return a.fileName < b.fileName end)

local ensemble = {}
if #opt.model > 1 then
    print('Ensemble!')
    for i = 1, #opt.model do
        print('\t' .. opt.model[i])
    end
end

local globalTimer = torch.Timer()
for i = 1, #opt.model do
    for modelFile in paths.iterfiles('model') do
        if modelFile:find('.t7') and modelFile:find(opt.model[i]) then
            local model = torch.load(paths.concat('model', modelFile)):cuda()
            if modelFile:find('multiscale') then
                print('This is multi-scale model! Swap the model')
                opt.scaleSwap = (opt.scaleSwap == -1) and (opt.scale - 1) or opt.scaleSwap
                model = scaleSwap(model)
            end
            local modelName = modelFile:split('%.')[1]
            print('>> Testing model: ' .. modelName)
            model:evaluate()

            local setTimer = torch.Timer()
            for j = 1, #testList do
                local localTimer = torch.Timer()
                if opt.progress then
                    io.write(('>>\t[%d/%d] %s ......'):format(j, #testList, testList[j].fileName))
                    io.flush()
                end

                local input = image.load(paths.concat(testList[j].from, testList[j].fileName), 3, 'float')
                local output = nil
                if opt.selfEnsemble then
                    output = util:x8Forward(input, model, opt.scale)
                else
                    local c, h, w = table.unpack(input:size():totable())
                    output = util:chopForward(input:cuda():view(1, c, h, w), model, opt.scale,
                        opt.chopShave, opt.chopSize)
                end

                if #opt.model > 1 then
                    if #ensemble < j then
                        table.insert(ensemble, output)
                    else
                        ensemble[i]:add(output)
                    end
                else
                    saveImage(testList[j], modelName, output)
                end

                input = nil
                target = nil
                output = nil
                collectgarbage()
                collectgarbage()

                if opt.progress then
                    io.write(('\tdone. (time: %.3fs)\n'):format(localTimer:time().real))
                    io.flush()
                end
            end
            local elapsed = timer:time().real
            print(('Elapsed time: %.3f (average %.3f)'):format(elapsed, elapsed / #testList))
        end
    end
end

if #opt.model > 1 then
    print('Averaging all the results...')
    for i = 1, #ensemble do
        ensemble[i]:div(#opt.model)
        saveImage(testList[i], table.concat(opt.model, '_', ensemble[i])
    end
    local elapsed = globalTimer:time.real
    print(('Elapsed time: %.3f (average %.3f)'):format(elapsed, elapsed / #testList))
end

local function scaleSwap(model)
    local sModel = nn.Sequential()
    sModel
        :add(model:get(1))
        :add(model:get(2))
        :add(model:get(3))
        :add(model:get(4):get(opt.scaleSwap))
        :add(model:get(5):get(opt.scaleSwap))

    return sModel:cuda()
end

local function saveImage(info, modelName, output)
    util:quantize(output, opt.mulImg)
    if opt.type == 'bench' or opt.type == 'val' then
        local target = image.load(paths.concat(dataDir, info.setName, info.fileName), 3, 'float')
        target = target[{{}, {1, output:size(2)}, {1, output:size(3)}}]
        image.save(paths.concat('img_target', modelName, info.setName, info.fileName), target)
        image.save(paths.concat('img_output', modelName, info.setName, Xs, info.fileName), output)
    elseif opt.type == 'test' then
        image.save(paths.concat('img_output', modelName, 'test', X2, info.fileName), output)
    end
end