local image = require 'image'
local paths = require 'paths'
local transform = require 'data/transforms'

local M = {}
local div2k = torch.class('sr.div2k', M)

function div2k:__init(opt, split)
    self.size = 800
    self.opt = opt
    self.split = split

    --absolute path of the dataset
    local apath = paths.concat(opt.datadir, 'dataset/DIV2K') -- '/var/tmp/dataset/DIV2K'
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
    if ext == '.png' then
        input = image.load(paths.concat(self.dirInp, inputName), self.opt.nChannel, 'float')
        target = image.load(paths.concat(self.dirTar, targetName), self.opt.nChannel, 'float')
    else
        input = torch.load(paths.concat(self.dirInp, inputName)):float():div(255)
        target = torch.load(paths.concat(self.dirTar, targetName)):float():div(255)
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

    input:mul(self.opt.mulImg)
    target:mul(self.opt.mulImg)

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

function div2k:augment()
    if self.split == 'train' then
        local transforms = {}
        if self.opt.colorAug then
            table.insert(transforms,
                transform.ColorJitter({
                    brightness = 0.1,
                    contrast = 0.1,
                    saturation = 0.1
                })
            )
        end
        table.insert(transforms, transform.HorizontalFlip(0.5))
        table.insert(transforms, transform.VerticalFlip(0.5))
        table.insert(transforms, transform.Rotation(1))

        return transform.Compose(transforms)
    elseif self.split == 'val' then
        return function(sample) return sample end
    end
end

return M.div2k
