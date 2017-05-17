local paths = require 'paths'
local transform = require 'data/transforms'

local M = {}
local imagenet50k = torch.class('sr.imagenet50k', M)

function imagenet50k:__init(opt, split)
    self.opt = opt
    self.split = split

    self.scale = self.opt.scale

    self.dataSize = 50000
    self.nVal = 5                   --We use Set5 for validation

    --Absolute path of the dataset
    local apath = nil

    assert(opt.degrade == 'bicubic', 'IMAGENET only supports bicubic degrader')
    if opt.datatype == 'png' then
        self.ext = '.png'
    elseif opt.datatype == 't7' then
        self.ext = '.t7'
    end

    if self.split == 'train' then
        if opt.datatype == 'png' then
            apath = paths.concat(opt.datadir, 'IMAGENET')
        elseif opt.datatype == 't7' then
            apath = paths.concat(opt.datadir, 'IMAGENET_decoded')
        end

        local tHR = 'IMAGENET_HR'
        local tLR = 'IMAGENET_LR_'

        self.dirTar = paths.concat(apath, tHR)
        self.dirInp = {}
        for i = 1, #self.scale do
            table.insert(self.dirInp, paths.concat(apath, tLR .. opt.degrade, 'X' .. self.scale[i]))
            self.dirInp[i] = self.dirInp[i]
        end
    elseif self.split == 'val' then
        apath = paths.concat(opt.datadir, 'benchmark')
        local tHR = 'Set5'
        local tLR = 'small/Set5'

        self.dirTar = paths.concat(apath, tHR)
        self.dirInp = {}
        for i = 1, #self.scale do
            table.insert(self.dirInp, paths.concat(apath, tLR, 'X' .. self.scale[i]))
        end
    end
    

    collectgarbage()
    collectgarbage()
end

function imagenet50k:get(idx, scaleIdx)
    local idx = idx
    local scale = self.scale[scaleIdx]
    local dataSize = self.dataSize

    local input, target = nil
    local inputName, targetName, rot = self:getFileName(idx, scale)
    if self.opt.datatype == 'png' or self.split == 'val' then
	    input = image.load(paths.concat(self.dirInp[scaleIdx], inputName), self.opt.nChannel, 'float')
        target = image.load(paths.concat(self.dirTar, targetName), self.opt.nChannel, 'float')    
    elseif self.opt.datatype == 't7' then
        input = torch.load(paths.concat(self.dirInp[scaleIdx], inputName))
        target = torch.load(paths.concat(self.dirTar, targetName))
    end

    local _, h, w = unpack(target:size():totable())
    local hInput, wInput = math.floor(h / scale), math.floor(w / scale)
    local hTarget, wTarget = scale * hInput, scale * wInput
    target = target[{{}, {1, hTarget}, {1, wTarget}}]
    
    local patchSize = self.opt.patchSize
    local targetPatch = self.opt.multiPatch and (patchSize * scale) or patchSize
    local inputPatch = targetPatch / scale

    if (wTarget < targetPatch) or (hTarget < targetPatch) then
        return nil
    end

    --Generate patches for training
    if self.split == 'train' then
        local ix = torch.random(1, wInput - inputPatch + 1)
        local iy = torch.random(1, hInput - inputPatch + 1)
        local tx = scale * (ix - 1) + 1
        local ty = scale * (iy - 1) + 1
        input = input[{{}, {iy, iy + inputPatch - 1}, {ix, ix + inputPatch - 1}}]
        target = target[{{}, {ty, ty + targetPatch - 1}, {tx, tx + targetPatch - 1}}]
    end

    if self.opt.datatype == 'png' or self.split == 'val' then
        input:mul(self.opt.mulImg)
        target:mul(self.opt.mulImg)
    else
        input = input:float():mul(self.opt.mulImg / 255)
        target = target:float():mul(self.opt.mulImg / 255)
    end

    --Reject the patch that has small size of spatial gradient
    if self.split == 'train' and self.opt.rejection ~= -1 then
        local grT, grP = nil, nil
        if self.opt.rejectionType == 'input' then
            grT, grP = input, inputPatch
        elseif self.opt.rejectionType == 'target' then
            grT, grP = target, targetPatch
        end

        local dx = grT[{{}, {1, grP - 1}, {1, grP - 1}}] - grT[{{}, {1, grP - 1}, {2, grP}}]
        local dy = grT[{{}, {1, grP - 1}, {1, grP - 1}}] - grT[{{}, {2, grP}, {1, grP - 1}}]
        local dsum = dx:pow(