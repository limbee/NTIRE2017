--This code ensembles every model you typed in

require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'optim'
require '../code/model/common'

local cmd = torch.CmdLine()
cmd:option('-type',     'val', 	        'demo type: bench | test | val')
cmd:option('-dataset',  'DIV2K',        'test dataset')
cmd:option('-dataSize', 'small',         'test data size')
cmd:option('-mulImg',   255,            'multiply constant to input image')
cmd:option('-progress', 'true',         'show current progress')
cmd:option('-ensemble', '',             'Models to ensemble.Delimiter is +')
cmd:option('-degrade',  'bicubic',      'degrading opertor: bicubic | unknown')
cmd:option('-scale',    2,              'scale factor: 2 | 3 | 4')
cmd:option('-gpuid',	1,		        'GPU id for use')
cmd:option('-datadir',	'/var/tmp',		'data directory')
cmd:option('-fr',       'false',        'enables self ensemble with flip and rotation')
cmd:option('-deps',     '',             'additional dependencies for testing')

local opt = cmd:parse(arg or {})
opt.progress = (opt.progress == 'true')
opt.fr = (opt.fr == 'true')

print('Ensemble:')
print(opt.ensemble)
opt.ensemble = opt.ensemble:split('+')
local modelName = table.concat(opt.ensemble, '_')

local now = os.date('%Y-%m-%d_%H-%M-%S')
local util = require '../code/utils'(nil)

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid)

local eO = {}
local dataDir = ''
local dataSize = opt.dataSize
local Xs = 'X' .. opt.scale

--testList[i][1]: absolute directory
--testList[i][2]: image file name
--testList[i][3]: benchmark set name
testList = {}

if (opt.type == 'bench') or (opt.type == 'val') then
    dataDir = paths.concat(opt.datadir, 'dataset/benchmark')
    for testFolder in paths.iterdirs(paths.concat(dataDir, dataSize)) do
        if (opt.type == 'bench') or ((opt.type == 'val') and (testFolder == 'val')) then
            local inputFolder = paths.concat(dataDir, dataSize, testFolder, Xs)
            paths.mkdir(paths.concat('img_output', modelName, testFolder, Xs))
            paths.mkdir(paths.concat('img_target', modelName, testFolder))
            for testFile in paths.iterfiles(inputFolder) do
                if testFile:find('.png') then
                    table.insert(testList, {inputFolder, testFile, testFolder})
                end
            end
        end
    end
elseif opt.type == 'test' then
    --This code is for DIV2K dataset
    if opt.dataset == 'DIV2K' then
        dataDir = paths.concat(opt.datadir, 'dataset/DIV2K/DIV2K_valid_LR_' .. opt.degrade, Xs)
        paths.mkdir(paths.concat('img_output', modelName, 'test', Xs))
        for testFile in paths.iterfiles(dataDir) do
            if testFile:find('.png') then
                table.insert(testList, {dataDir, testFile})
            end
        end
    --you can test with our own images.
    --Just put the images in the img_input folder.
    else
        for testFile in paths.iterfiles('img_input') do
            if testFile:find('.png') or testFile:find('.jp') then
                table.insert(testList, {'img_input', testFile})
            end
        end
    end
end
table.sort(testList, function(a,b) return a[2] < b[2] end)

local globalTimer = torch.Timer()
for i = 1, #opt.ensemble do
    local model = torch.load(paths.concat('model', opt.ensemble[i] .. '.t7')):cuda()
    print('>> Testing model: ' .. opt.ensemble[i])
    model:evaluate()

    local timer = torch.Timer()
    for i = 1, #testList do
        local timerLocal = torch.Timer()
        if opt.progress then
            io.write(('>> \t [%d/%d] %s ......'):format(i, #testList, testList[i][2]))
            io.flush()
        end

        local input = image.load(paths.concat(testList[i][1], testList[i][2]), 3, 'float'):mul(opt.mulImg)
        local output = nil
        if opt.fr then
            output = util:x8Forward(input, model)
        else
            local c, h, w = table.unpack(input:size():totable())
            output = util:recursiveForward(input:cuda():view(1, c, h, w), model):squeeze(1)
        end

        if #eO < i then
            table.insert(eO, output)
        else
            eO[i] = eO[i] + output
        end

        input = nil
        output = nil
        collectgarbage()
        collectgarbage()

        if opt.progress then
            io.write(('\t done. (time: %.2fs) \n'):format(timerLocal:time().real))
            io.flush()
        end
    end
    local elapsed = timer:time().real
    print(('Elapsed time: %.2f (Average %.2f)'):format(elapsed, elapsed / #testList))
end

print('Averaging all the results...')
for i = 1, #eO do
    eO[i]:div(#opt.ensemble)
    util:quantize(eO[i], opt.mulImg)
    if (opt.type == 'bench') or (opt.type == 'val') then
        local target = image.load(paths.concat(dataDir, testList[i][3], testList[i][2]), 3, 'float')
        target = target[{{}, {1, eO[i]:size(2)}, {1, eO[i]:size(3)}}]
        image.save(paths.concat('img_target', modelName, testList[i][3], testList[i][2]), target)
        image.save(paths.concat('img_output', modelName, testList[i][3], Xs, testList[i][2]), eO[i])
    elseif opt.type == 'test' then
        image.save(paths.concat('img_output', modelName, 'test', Xs, testList[i][2]), eO[i])
    end
    
end
local gT = globalTimer:time().real
print(('Total time: %.2f (Average %.2f)'):format(gT, gT / #testList))