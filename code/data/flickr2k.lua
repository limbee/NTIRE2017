local paths = require 'paths'
local transform = require 'data/transforms'

local M = {}
local flickr2k = torch.class('sr.flickr2k', M)

function flickr2k:__init(opt, split)
    self.opt = opt
    self.split = split

    self.size = opt.nTrain_Flickr2K
    self.nDIV2K = opt.nTrain_DIV2K
    self.offset = opt.valOffset
    self.nVal = opt.nVal
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

    self.dirTar_Flickr2K = paths.concat(apath, 'Flickr2K_HR')
    self.dirInp_Flickr2K = {}
    for i = 1, #self.scale do
        local orgPath = paths.concat(apath, 'Flickr2K_LR_' .. opt.degrade, 'X' .. self.scale[i])
        local augPath = paths.concat(apath, 'Flickr2K_LR_' .. opt.degrade .. '_augment', 'X' .. self.scale[i])
        
        if opt.augUnkFlick2K and not paths.dirp(augPath)then
            error('Augmented DIV2K LR unknown dataset does not exist.')
        end

        if not opt.augUnkFlickr2K then
            table.insert(self.dirInp_Flickr2K, orgPath)
        else
            table.insert(self.dirInp_Flickr2K, augPath)
        end
    end

    if opt.datatype == 'png' then
        apath = paths.concat(opt.datadir, 'DIV2K')
    elseif opt.datatype == 't7' then
        apath = paths.concat(opt.datadir, 'DIV2K_decoded')
    end
    self.dirTar_DIV2K = paths.concat(apath, 'DIV2K_train_HR')
    self.dirInp_DIV2K_org = {}
    self.dirInp_DIV2K_aug = {}

    for i = 1, #self.scale do
        local orgPath = paths.concat(apath, 'DIV2K_train_LR_' .. opt.degrade, 'X' .. self.scale[i])
        local augPath = paths.concat(apath, 'DIV2K_train_LR_' .. opt.degrade .. '_augment', 'X' .. self.scale[i])

        table.insert(self.dirInp_DIV2K_org, orgPath)
        table.insert(self.dirInp_DIV2K_aug, augPath)
    end
    collectgarbage()
    collectgarbage()
end

function flickr2k:get(idx, scaleIdx)
    local idx = idx
    local scale = self.scale[scaleIdx]
    local dataSize = self.dataSize

    local function getImg(idx, scale, type)
        local dirInp, dirTar, nDigit
        if type == 'DIV2K' then
            if self.split == 'train' and self.opt.augUnkDIV2K then
                dirInp = self.dirInp_DIV2K_aug[scaleIdx]
            else
                dirInp = self.dirInp_DIV2K_org[scaleIdx]
            end
            dirTar = self.dirTar_DIV2K
            nDigit = 4
        elseif type == 'Flickr2K' then
            dirInp = self.dirInp_Flickr2K[scaleIdx]
            dirTar = self.dirTar_Flickr2K
            nDigit = 6
        end

        if idx > self.opt.flickr2kSize then
            idx = idx - self.opt.flickr2kSize
            if idx > self.offset then
                idx = idx + self.nVal
            end
        end

        local inputName, targetName, rot = self:getFileName(idx, scale, nDigit, type)
        if self.opt.datatype == 't7' then
            input = torch.load(paths.concat(dirInp, inputName))
            target = torch.load(paths.concat(dirTar, targetName))
        elseif self.opt.datatype == 'png' then
            input = image.load(paths.concat(dirInp, inputName), self.opt.nChannel, 'float')
            target = image.load(paths.concat(dirTar, targetName), self.opt.nChannel, 'float')
        end
        return input, target, rot
    end

    local rot = 1

    if self.split == 'val' then
        idx = idx + self.offset
        input, target = getImg(idx, scale, 'DIV2K')
    else
        if not self.opt.useDIV2K then
            input, target = getImg(idx, scale, 'Flickr2K')
        else
            if idx <= self.opt.flickr2kSize then
                input, target, rot = getImg(idx, scale, 'Flickr2K')
            else -- then loads DIV2K
                input, target, rot = getImg(idx, scale, 'DIV2K')
            end
        end
    end

    if rot == 1 then
        target = target
    elseif rot == 2 then
        target = image.vflip(target)
    elseif rot == 3 then
        target = image.hflip(target)
    elseif rot == 4 then
        target = image.hflip(image.vflip(target))
    elseif rot == 5 then
        target = target:transpose(2,3)
    elseif rot == 6 then
        target = (image.vflip(target)):transpose(2,3)
    elseif rot == 7 then
        target = (image.hflip(target)):transpose(2,3)
    elseif rot == 8 then
        target = (image.hflip(image.vflip(target))):transpose(2,3)
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
        if self.opt.rejectionType == 'input' then
            grT, grP = input, inputPatch
        elseif self.opt.rejectionType == 'target' then
            grT, grP = target, targetPatch
        end

        local dx = grT[{{}, {1, grP - 1}, {1, grP - 1}}] - grT[{{}, {1, grP - 1}, {2, grP}}]
        local dy = grT[{{}, {1, grP - 1}, {1, grP - 1}}] - grT[{{}, {2, grP}, {1, grP - 1}}]
        local dsum = dx:pow(2) + dy:pow(2)
        local dsqrt = dsum:sqrt()
        local gradValue = dsqrt:view(-1):mean()
        
        if self.gradStatistics == nil then
            self.gradSamples = self.opt.nGradStat
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
        if self.opt.useDIV2K then
            return self.size + self.nDIV2K - self.nVal
        else
            return self.size
        end
    elseif self.split == 'val' then
        return self.nVal
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

function flickr2k:getFileName(idx, scale, nDigit, type)
    --filename format: ??????x?.png
    local nDigit = nDigit or 6
    local fileName = idx
    local digit = idx
    while digit < math.pow(10, nDigit - 1) do
        fileName = '0' .. fileName
        digit = digit * 10
    end

    local targetName = fileName .. self.ext
    local inputName = nil

    local rot
    if self.split == 'train' and ((type == 'DIV2K' and self.opt.augUnkDIV2K) or (type == 'Flickr2K' and self.opt.augUnkFlickr2K)) then
        rot = math.random(1,8)
        inputName = fileName .. 'x' .. scale .. '_' .. rot .. self.ext
    else
        rot = 1
        inputName = fileName .. 'x' .. scale .. self.ext
    end

    return inputName, targetName, rot
end

return M.flickr2k
