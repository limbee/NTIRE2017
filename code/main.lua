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
local load, loss, psnr, startEpoch = util:load()

local DataLoader = require 'dataloader'
local Trainer = require 'train'

print('loading model and criterion...')
local model = require 'model/init'(opt, startEpoch)
local criterion = require 'loss/init'(opt)

print('Creating data loader...')
local trainLoader, valLoader = DataLoader.create(opt)
local trainer = Trainer(model, criterion, opt)

if opt.testOnly == 'true' then
    print('Test Only')
    trainer:test(startEpoch, valLoader)

elseif opt.testOnly == 'flase' then
    print('Train start')
    for epoch = startEpoch, opt.nEpochs do
        loss[epoch] = trainer:train(epoch, trainLoader)
        psnr[epoch] = trainer:test(epoch, valLoader)

        util:plot(loss,'loss')
        util:plot(psnr,'PSNR')
        util:checkpoint(model, criterion, loss, psnr)
    end
else
    print("opt.testOnly is not valid type")
end