local image = require 'image'
local paths = require 'paths'
local transform = require 'data/transforms'
local util = require 'utils'()

local M = {}
local div2k = torch.class('sr.div2k', M)

function div2k:__init(opt, split)
    self.size = 800
    self.opt = opt
    self.split = split

    --absolute path of the dataset
    local apath = '/var/tmp/dataset/DIV2K'
    self.dirTar = paths.concat(apath, 'DIV2K_train_HR')
    self.dirInp = paths.concat(apath, 'DIV2K_train_LR_' .. opt.degrade, 'X' .. opt.scale)
    if opt.dataSize == 'big' then
        self.dirInp = self.dirInp .. 'b'
    end
end

function div2k:get(i)
    local netType = self.opt.netType
    local dataSize = self.opt.dataSize
    local idx = i
    if self.split == 'val' then
        idx = idx + (self.size - self.opt.numVal)
    end

    local scale = self.opt.scale
    local input = nil
    local target = nil

    --filename format: ????x?.png
    local fileName = idx
    local digit = idx
    while (digit < 1000) do
        fileName = '0' .. fileName
        digit = digit * 10
    end
    local ext = (self.opt.datatype == 'png') and '.png' or '.t7'
    inputName = fileName .. 'x' .. scale .. ext
    targetName = fileName .. ext
    if ext == 'png' then
        input = image.load(paths.concat(self.dirInp, inputName), self.opt.nChannel, 'float')
        target = image.load(paths.concat(self.dirTar, targetName), self.opt.nChannel, 'float')
    else
        input = torch.load(paths.concat(self.dirInp, inputName)):float()
        target = torch.load(paths.concat(self.dirTar, targetName)):float()
    end

    local channel, h, w = table.unpack(target:size():totable())
    local hInput, wInput = math.floor(h / scale), math.floor(w / scale)
    local hTarget, wTarget = scale * hInput, scale * wInput
    if dataSize == 'big' then
        hInput, wInput = hTarget, wTarget
    end
    target = target[{{}, {1, hTarget}, {1, wTarget}}]

    if self.split == 'train' then 
        local patchSize = self.opt.patchSize
        local targetPatch = patchSize
        local inputPatch = (dataSize == 'big') and patchSize or (patchSize / scale)
        if (wTarget < targetPatch) or (hTarget < targetPatch) then
            return
        end

        local ix = torch.random(1, wInput - inputPatch + 1)
        local iy = torch.random(1, hInput - inputPatch + 1)
        local tx, ty = (scale * (ix - 1)) + 1, (scale * (iy - 1)) + 1
        if dataSize == 'big' then
            tx, ty = ix, iy
        end
        input = input[{{}, {iy, iy + inputPatch - 1}, {ix, ix + inputPatch - 1}}]
        target = target[{{}, {ty , ty + targetPatch - 1}, {tx, tx + targetPatch - 1}}]
    end

    if self.opt.datatype == 'png' then
        input:mul(255)
        target:mul(255)
    end

    if self.opt.nChannel == 1 then
        input = util:rgb2y(input)
        target = util:rgb2y(target)
    end

    return {
        input = input,
        target = target
    }
end

function div2k:__size()
    if self.split == 'train' then
        return self.size - self.opt.numVal
    elseif self.split == 'val' then
        return self.opt.numVal
    end
end

-- Computed from random subset of ImageNet training images
local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
}
local pca = {
    eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
    eigvec = torch.Tensor{
        { -0.5675,  0.7192,  0.4009 },
        { -0.5808, -0.0045, -0.8140 },
        { -0.5836, -0.6948,  0.4203 },
    },
}

function div2k:augment()
    if self.split == 'train' then
        return transform.Compose{
            --[[
            transform.ColorJitter({
                brightness = 0.1,
                contrast = 0.1,
                saturation = 0.1
            }),
            --]]
            --transform.Lighting(0.1, pca.eigval, pca.eigvec),
            transform.HorizontalFlip(0.5),
            transform.Rotation(1)
        }
    elseif self.split == 'val' then
        return function(sample) return sample end
    end
end

return M.div2k
