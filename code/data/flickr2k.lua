local paths = require 'paths'
local transform = require 'data/transforms'

local M = {}
local flickr2k = torch.class('sr.flickr2k', M)

function flickr2k:__init(opt, split)
    self.opt = opt
    self.split = split

    self.size = 1e4 -- it would be changed
    self.offset = 1e4 + 790
    self.numVal = opt.numVal
    self.scale = self.opt.scale

    local apath = nil
    self.ext = nil

    if opt.datatype == 'png' then
        apath = paths.concat(opt.datadir, 'Flickr2K')
        self.ext = '.png'
    elseif opt.datatype == 't7' then
        apath = paths.concat(opt.datadir, 'Flickr2K_decoded')
        self.ext = '.t7'
    elseif opt.datatype == 't7pack' then
        error('No t7pack support. Flickr2K is too large.')
    else
        error('unknown -datatype (png | t7(default) | t7pack)')
    end

    local tHR = 'Flickr2K_HR'
    local tLR = 'Flickr2K_LR_'

    self.dirTar = paths.concat(apath, tHR)
    self.dirInp = {}

    for i = 1, #self.scale do
        table.insert(self.dirInp, paths.concat(apath, tLR .. opt.degrade, 'X' .. self.scale[i]))
    end

    collectgarbage()
    collectgarbage()
end

function flickr2k:get(idx, scaleIdx)
    local idx = idx
    local scale = self.scale[scaleIdx]
    local dataSize = self.dataSize

    if self.split == 'train' then
        if idx > self.offset then
            idx = idx + self.numVal
        end
    elseif self.split == 'val' then
        idx = idx + self.offset
    end

    local inputName, targetName = self:getFileName(idx, scale)
    if self.opt.datatype == 't7' then
        input = torch.load(paths.concat(self.dirInp[scaleIdx], inputName))
        target = torch.load(paths.concat(self.dirTar, targetName))
    elseif self.opt.datatype == 'png' then
        input = image.load(paths.concat(self.dirInp[scaleIdx], inputName), self.opt.nChannel, 'float')
        target = image.load(paths.concat(self.dirTar, targetName), self.opt.nChannel, 'float')
    end

    local _, h, w = unpack(target:size():totable())
    local hInput, wInput = math.floor(h / scale), math.floor(w / scale)
    local hTarget, wTarget = scale * hInput, scale * wInput
    target = target[{{}, {1, hTarget}, {1, wTarget}}]
    local patchSize = self.opt.patchSize
    local targetPatch = patchSize
    local inputPatch = patchSize / scale
    if wTarget < targetPatch or hTarget < targetPatch then
        return nil
    end

    if self.split == 'train' then
        local ix = torch.random(1, wInput - inputPatch + 1)
        local iy = torch.random(1, hInput - inputPatch + 1)
        local tx, ty = scale * (ix - 1) + 1, scale * (iy - 1) + 1
        input = input[{{}, {iy, iy + inputPatch - 1}, {ix, ix + inputPatch - 1}}]
        target = target[{{}, {ty, ty + targetPatch - 1}, {tx, tx + targetPatch - 1}}]
    end

    if self.opt.datatype == 'png' then
        input:mul(self.opt.mulImg)
        target:mul(self.opt.mulImg)
    else
        input = input:float():mul(self.opt.mulImg / 255)
        target = target:float():mul(self.opt.mulImg / 255)
    end

    --Reject the patch that has small size of spatial gradient
    if self.split == 'train' and self.opt.rejection ~= -1 then
        local grT, grP = nil, nil
        if self.opt.rejectionTarget == 'input' then
            grT, grP = input, inputPatch
        elseif self.opt.rejectionTarget == 'target' then
            grT, grP = target, targetPatch
        end

        local dx = grT[{{}, {1, grP - 1}, {1, grP - 1}}] - grT[{{}, {1, grP - 1}, {2, grP}}]
        local dy = grT[{{}, {1, grP - 1}, {1, grP - 1}}] - grT[{{}, {2, grP}, {1, grP - 1}}]
        local dsum = dx:pow(2) + dy:pow(2)
        local dsqrt = dsum:sqrt()
        local gradValue = dsqrt:view(-1):mean()
        
        if self.gradStatistics == nil then
            self.gradSamples = 10000
            self.gsTable = {}
            self.gradStatistics = {}
            for i = 1, #self.scale do
                table.insert(self.gsTable, {})
                table.insert(self.gradStatistics, -1)
            end
            print('Caculating median of gradient for ' .. self.gradSamples .. ' samples...')
            return nil
        end
        
        if #self.gsTable[scaleIdx] < self.gradSamples then
            table.insert(self.gsTable[scaleIdx], gradValue)
            return nil
        else
            if self.gradStatistics[scaleIdx] == -1 then
                local threshold = math.floor(self.gradSamples * self.opt.rejection / 100)
                table.sort(self.gsTable[scaleIdx])
                self.gradStatistics[scaleIdx] = self.gsTable[scaleIdx][threshold] / self.opt.mulImg
                print('Gradient threshold for scale ' .. self.scale[scaleIdx] .. ': ' .. self.gradStatistics[scaleIdx])
                return nil
            else
                if gradValue <= self.gradStatistics[scaleIdx] then
                    return nil
                end
            end
        end
    end

    return {
        input = input,
        target = target
    }
end

function flickr2k:__size()
    if self.split == 'train' then
        return self.size - self.numVal
    elseif self.split == 'val' then
        return self.numVal
    end
end

function flickr2k:augment()
    if self.split == 'train' and self.opt.degrade == 'bicubic' then
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
        -- We don't need vertical flip, since hflip + rotation covers it
        table.insert(transforms, transform.HorizontalFlip())
        table.insert(transforms, transform.Rotation())

        return transform.Compose(transforms)
    else
        return function(sample) return sample end
    end
end

function flickr2k:getFileName(idx, scale)
    --filename format: ??????x?.png
    local fileName = idx
    local digit = idx
    while digit < 1e5 do
        fileName = '0' .. fileName
        digit = digit * 10
    end

    local inputName = fileName .. 'x' .. scale .. self.ext
    local targetName = fileName .. self.ext

    return inputName, targetName
end

return M.flickr2k
