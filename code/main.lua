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

if opt.testOnly then
    print('Test Only')
    trainer:test(opt.startEpoch - 1, valLoader)
else
    print('Train start')
    local maxPerf = -1
    local maxIdx = -1
    for epoch = opt.startEpoch, opt.nEpochs do
        loss[epoch] = trainer:train(epoch, trainLoader)
        trainer:reTrain()
        psnr[epoch] = trainer:test(epoch, valLoader)
        
        if psnr[epoch] > maxPerf then
            maxPerf = psnr[epoch]
            maxIdx = epoch
        end
        print(string.format('Highest PSNR: %.4f (epoch %d)\n', maxPerf, maxIdx))

        util:plot(loss,'loss')
        util:plot(psnr,'PSNR')
        util:checkpoint(model, criterion, loss, psnr)
    end
end
