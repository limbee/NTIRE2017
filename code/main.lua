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
local load, loss, psnr, lr = util:load()

local DataLoader = require 'dataloader'
local Trainer = require 'train'

print('loading model and criterion...')
local model = require 'model/init'(opt)
local criterion = require 'loss/init'(opt)

print('Creating data loader...')
local trainLoader, valLoader = DataLoader.create(opt)
local trainer = Trainer(model, criterion, opt)

if opt.valOnly then
    print('Validate the model (at epoch ' .. opt.startEpoch - 1 .. ') with ' .. opt.nVal .. ' val images')
    trainer:test(opt.startEpoch - 1, valLoader)
else
    print('Train start')

    for epoch = opt.startEpoch, opt.nEpochs do
        trainer:train(epoch, trainLoader)
        util:plot(trainer:updateLoss(loss), 'Loss')
        util:plot(trainer:updateLR(lr), 'Learning Rate')
        
        --Skip validation
        if not opt.trainOnly then
            trainer:test(epoch, valLoader)
            util:plot(trainer:updatePSNR(psnr), 'PSNR', opt.psnrLabel)
        end
        
        util:checkpoint(model, criterion, loss, psnr, lr)
    end
end
