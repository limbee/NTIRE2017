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

print('Train start')
local startEpoch = load and #loss+1 or opt.epochNumber
for epoch = startEpoch, opt.nEpochs do
    local loss_ = trainer:train(epoch, trainLoader)
    local psnr_ = trainer:test(epoch, valLoader)

    loss[#loss+1] = loss_
    psnr[#psnr+1] = psnr_

    util:plot(loss,'loss')
    util:plot(psnr,'PSNR (' .. opt.valset ..')')

    util:store(model,criterion,loss,psnr)
end