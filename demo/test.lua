require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
local threads = require 'threads'
require '../code/model/common'

local cmd = torch.CmdLine()
cmd:option('-type',         'val', 	    'demo type: bench | test | val')
cmd:option('-dataset',      'DIV2K',    'test dataset')
cmd:option('-mulImg',       255,        'multiply constant to input image')
cmd:option('-progress',     'true',     'show current progress')
cmd:option('-model',        'resnet',   'substring of model name')
cmd:option('-save',         '.',        'Save as')
cmd:option('-ensembleW',    '-1',        'Ensemble weight')
cmd:option('-degrade',      'bicubic',  'degrading opertor: bicubic | unknown')
cmd:option('-scale',        2,          'scale factor: 2 | 3 | 4')
cmd:option('-swap',         -1,         'Model swap')
cmd:option('-gpuid',	    1,		    'GPU id for use')
cmd:option('-nThreads',     3,          'Number of threads to save images')
cmd:option('-nGPU',         1,          'Number of GPUs to use by default')
cmd:option('-dataDir',	    '/var/tmp', 'data directory')
cmd:option('-selfEnsemble', 'false',    'enables self ensemble with flip and rotation')
cmd:option('-chopShave',    10,         'Shave width for chopForward')
cmd:option('-chopSize',     16e4,       'Minimum chop size for chopForward')
cmd:option('-inplace',      'false',    'inplace operation')

local opt = cmd:parse(arg or {})
opt.progress = (opt.progress == 'true')
opt.model = opt.model:split('+')
opt.ensembleW = opt.ensembleW:split('_')
for i = 1, #opt.ensembleW do
    opt.ensembleW[i] = tonumber(opt.ensembleW[i])
end
if #opt.ensembleW > 1 then

end
opt.selfEnsemble = (opt.selfEnsemble == 'true')

local util = require '../code/utils'(opt)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid)

local pool = threads.Threads(
    opt.nThreads,
    function(threadid)
        print('Starting a background thread...')
        require 'cunn'
        require 'image'
    end
)

--Prepare the dataset for demo
print('Preparing dataset...')
local testList = {}
local dataDir = ''
local Xs = 'X' .. opt.scale

local function swap(model)
    local sModel = nn.Sequential()
    sModel
        :add(model:get(1))
        :add(model:get(2))
        :add(model:get(3))
        :add(model:get(4):get(opt.swap))
        :add(model:get(5):get(opt.swap))

    return sModel:cuda()
end

local function saveImage(info, modelName)
    info.saveImg:mul(255 / opt.mulImg):add(0.5):floor():div(255)
    info.saveImg = info.saveImg:squeeze(1)
    modelName = (opt.save == '.') and modelName or opt.save
    if opt.type == 'bench' or opt.type == 'val' then
        local targetDir = paths.concat('img_target', modelName, info.setName)
        local outputDir = paths.concat('img_output', modelName, info.setName, Xs)
        if not paths.dirp(targetDir) then
            paths.mkdir(targetDir)
        end
        if not paths.dirp(outputDir) then
            paths.mkdir(outputDir)
        end
        local targetFrom = paths.concat(dataDir, info.setName, info.fileName)
        local targetTo = paths.concat(targetDir, info.fileName)
        os.execute('cp ' .. targetFrom .. ' ' .. targetTo)
        image.save(paths.concat(outputDir, info.fileName), info.saveImg)
    elseif opt.type == 'test' then
        local outputDir = paths.concat('img_output', modelName, info.setName, Xs)
        if not paths.dirp(outputDir) then
            paths.mkdir(outputDir)
        end
        image.save(paths.concat(outputDir, info.fileName), info.saveImg)
    end

    info.saveImg = nil
    collectgarbage()
    collectgarbage()
end

local function makeinplace(model)

    if model.modules then
        for i = 1, #model.modules do
            if model:get(i).inplace == false then
                -- print(model:get(i).inplace)
                model:get(i).inplace = true
                -- print(model:get(i))
                -- print(model:get(i).inplace)
            end

            if model:get(i).modules then
                makeinplace(model:get(i))
            end
        end
    end

    -- require 'trepl'()

    return model

end

local loadTimer = torch.Timer()
if opt.type == 'bench' or opt.type == 'val' then
    dataDir = paths.concat(opt.dataDir, 'dataset/benchmark')
    for benchFolder in paths.iterdirs(paths.concat(dataDir, 'small')) do
        if opt.type == 'bench' or (opt.type == 'val' and benchFolder == 'val') then
            local inputFolder = paths.concat(paths.concat(dataDir, 'small'), benchFolder, Xs)
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
        dataDir = paths.concat(opt.dataDir, 'dataset/DIV2K/DIV2K_valid_LR_' .. opt.degrade, Xs)
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
for i = 1, #testList do
    testList[i].inputImg = image.load(paths.concat(testList[i].from, testList[i].fileName), 3, 'float'):mul(opt.mulImg)
end
local loadElapsed = loadTimer:time().real
print(('[Load time] %.3fs (average %.3fs)\n'):format(loadElapsed, loadElapsed / #testList))

table.sort(testList, function(a, b) return a.fileName < b.fileName end)

local ensemble = {}
if #opt.model > 1 then
    print('Ensemble!')
    for i = 1, #opt.model do
        print('\t' .. opt.model[i])
    end
    print('')
end

local globalTimer = torch.Timer()
local nModel = 0
local totalModelName = {}
if #opt.model > 1 then
    for i = 1, #opt.model do
        table.insert(totalModelName, opt.model[i]:split('%.')[1])
    end
end
for i = 1, #opt.model do
    for modelFile in paths.iterfiles('model') do
        if modelFile:find('.t7') and modelFile:find(opt.model[i]) then
            local model = torch.load(paths.concat('model', modelFile))
            model = makeinplace(model):cuda()
            
            local modelName = modelFile:split('%.')[1]
            print('Model: [' .. modelName .. ']')
            if modelFile:find('multiscale') then
                print('This is a multi-scale model! Swap the model')
                opt.swap = (opt.swap == -1) and (opt.scale - 1) or opt.swap
                model = swap(model)
            end

            if opt.nGPU > 1 then
                local gpus = torch.range(1, opt.nGPU):totable()
                local dpt = nn.DataParallelTable(1, true, true)
                    :add(model, gpus)
                model = dpt:cuda()
            end
            model:evaluate()

            local setTimer = torch.Timer()
            for j = 1, #testList do
                local localTimer = torch.Timer()
                if opt.progress then
                    io.write(('>> [%d/%d]\t%s\t'):format(j, #testList, testList[j].fileName))
                    io.flush()
                end

                local input = testList[j].inputImg
                local output = nil
                if opt.selfEnsemble then
                    output = util:x8Forward(input, model, opt.scale, opt.nGPU)
                else
                    local c, h, w = table.unpack(input:size():totable())
                    output = util:chopForward(input:cuda():view(1, c, h, w), model, opt.scale,
                        opt.chopShave, opt.chopSize, opt.nGPU)
                end

                if #opt.model > 1 then
                    local ensembleWeight = (#opt.ensembleW) == 1 and 1 or opt.ensembleW[i] * #opt.model
                    if #ensemble < j then
                        table.insert(ensemble, output:mul(ensembleWeight))
                    else
                        ensemble[j]:add(output:mul(ensembleWeight))
                    end
                else
                    testList[j].saveImg = output:float()
                end
                if (#opt.model == 1) or ((#opt.model > 1) and (i == #opt.model)) then
                    if #opt.model > 1 then
                        modelName = table.concat(totalModelName, '_')
                        testList[j].saveImg = ensemble[j]:div(#opt.model)
                    end
                    pool:addjob(
                        function()
                            saveImage(testList[j], modelName)
                            return __threadid
                        end,
                        function(id)
                        end
                    )
                end
                input = nil
                target = nil
                output = nil
                model:clearState()
                collectgarbage()
                collectgarbage()

                if opt.progress then
                    io.write(('[time] %.3fs\n'):format(localTimer:time().real))
                    io.flush()
                end
            end
            local elapsed = setTimer:time().real
            print(('[Forward time] %.3fs (average %.3fs)'):format(elapsed, elapsed / #testList))
            local saveTimer = torch.Timer()
            pool:synchronize()
            local saveElapsed = saveTimer:time().real
            print(('[Save time] %.3fs (average %.3fs)'):format(saveElapsed, saveElapsed / #testList))
            nModel = nModel + 1
            print('')
        end
    end
end
pool:terminate()

local totalElapsed = globalTimer:time().real
print(('[Total time] %.3fs (average %.3fs)'):format(totalElapsed, totalElapsed / #testList))
