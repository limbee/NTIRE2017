local paths = require 'paths'
local transform = require 'data/transforms'

local M = {}
local mnist = torch.class('sr.mnist', M)

function mnist:__init(opt, split)
    self.opt = opt
    self.split = split

    self.nTrain = 59000
    self.nVal = 60000 - self.nTrain
    self.scale = self.opt.scale[1]

    --Absolute path of the dataset
    local apath = paths.concat(opt.datadir, 'MNIST')

    local input = torch.load(paths.concat(apath, 'train_x' .. self.scale .. '.t7'))
    local target = torch.load(paths.concat(apath, 'train.t7'))

    if split == 'train' then
        self.t7Inp = input[{{1, self.nTrain}}]:clone()
        self.t7Tar = target[{{1, self.nTrain}}]:clone()
    elseif split == 'val' then
        self.t7Inp = input[{{self.nTrain + 1, -1}}]:clone()
        self.t7Tar = target[{{self.nTrain + 1, -1}}]:clone()
    end

    input = nil
    target = nil

    collectgarbage()
    collectgarbage()
end

function mnist:get(idx, scaleIdx)
    local input = self.t7Inp[idx]:float()
    local target = self.t7Tar[idx]:float()
   
    return {
        input = input,
        target = target
    }
end

function mnist:__size()
    if self.split == 'train' then
        return self.nTrain
    elseif self.split == 'val' then
        return self.nVal
    end
end

function mnist:augment()
    return function(sample) return sample end
end

return M.mnist

