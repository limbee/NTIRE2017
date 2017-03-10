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
if load then
    local prevlr = opt.lr
    opt.lr = opt.lr / math.pow(2, math.floor(#loss / opt.manualDecay)
    print('Learning rate decreased: ' prevlr .. ' -> ' .. opt.lr)
end

local DataLoader = require 'dataloader'
local Trainer = require 'train'

print('loading model and criterion...')
local model = require 'model/init'(opt)
local criterion = require 'loss/init'(opt)

print('Creating data loader...')
local trainLoader, valLoader = DataLoader.create(opt)
local trainer = Trainer(model, criterion, opt)

print('Train start')
local startEpoch = load and #loss + 1 or opt.epochNumber
for epoch = startEpoch, opt.nEpochs do
    loss[#loss + 1] = trainer:train(epoch, trainLoader)
    psnr[#psnr + 1] = trainer:test(epoch, valLoader)

    util:plot(loss,'loss')
    util:plot(psnr,'PSNR')

    util:checkpoint(model, criterion, loss, psnr)
end
