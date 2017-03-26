require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cutorch'
require 'gnuplot'

print('\n\n\n' .. os.date("%Y-%m-%d_%H-%M-%S") .. '\n')
local opts = require 'opts'
local opt = opts.parse(arg)

local util = require 'utils'(opt)
local load, loss, psnr = util:load()

local DataLoader = require 'dataloader'
local Trainer = require 'train'

print('loading model and criterion...')
local model = require 'model/init'(opt)
local criterion = require 'loss/init'(opt)

print('Creating data loader...')
local trainLoader, valLoader = DataLoader.create(opt)
local trainer = Trainer(model, criterion, opt)

local scale = nil
local label = nil
if type(opt.scale) == 'number' then
    scale = {opt.scale}
    label = {'PSNR (X' .. opt.scale .. ')'}
else
    scale = opt.scale:split('|')
    for i = 1, #scale do
        scale[i] = tonumber(scale[i])
        table.insert(label, 'PSNR (X' .. scale[i] .. ')')
    end
end

if opt.testOnly then
    print('Test Only')
    trainer:test(opt.startEpoch - 1, valLoader)
else
    print('Train start')
    --Code for multiscale learning
    maxPerf, maxIdx = {}, {}
    for i = 1, #scale do
        table.insert(maxPerf, -1)
        table.insert(maxIdx, -1)
    end

    for epoch = opt.startEpoch, opt.nEpochs do
        loss[epoch] = trainer:train(epoch, trainLoader)
        trainer:reTrain()
        psnr[epoch] = trainer:test(epoch, valLoader)
        
        --Code for multiscale learning
        for i = 1, #scale do
            if psnr[epoch][i] > maxPerf[i] then
                maxPerf[i] = psnr[epoch][i]
                maxIdx[i] = epoch
            end
            print(('Highest PSNR: %.4f (X%d) - epoch %d\n')
                :format(maxPerf[i], scale[i], maxIdx[i]))
        end

        util:plot(loss, 'loss')
        util:plot(psnr, 'PSNR', label)
        util:checkpoint(model, criterion, loss, psnr)
    end
end
