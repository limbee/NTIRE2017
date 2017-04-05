
local paths = require 'paths'
local transform = require 'data/transforms'

local M = {}
local div2k = torch.class('sr.div2k', M)

function div2k:__init(opt, split)
    self.opt = opt
    self.split = split

    self.size = 800
    self.offset = self.size - self.opt.numVal
    self.scale = self.opt.scale
    self.dataSize = self.opt.dataSize

    --Absolute path of the dataset
    local apath = nil
    self.ext = nil

    if opt.datatype == 'png' then
        apath = paths.concat(opt.datadir, 'DIV2K')
        self.ext = '.png'
    else
        apath = paths.concat(opt.datadir, 'DIV2K_decoded')
        self.ext = '.t7'
    end

    local tHR = 'DIV2K_train_HR'
    local tLR = 'DIV2K_train_LR_'

    self.dirTar = paths.concat(apath, tHR)
    self.dirInp = {}

    for i = 1, #self.scale do
        table.insert(self.dirInp, paths.concat(apath, tLR .. opt.degrade, 'X' .. self.scale[i]))
        self.dirInp[i] = opt.dataSize == 'small' and self.dirInp[i] or self.dirInp[i]
        self.dirInp[i] = opt.netType ~= 'recurVDSR' and self.dirInp[i] or self.dirInp[i] .. '_SRresOutput'
    end

    --Load single .t7 files that contains all dataset
    if opt.datatype == 't7pack' then
        print('\tLoading t7pack:')
        if split == 'train' then
            --Here, we will split the validation sets and save them as *v.t7 file
            self.t7Tar = torch.load(paths.concat(self.dirTar, 'pack.t7'))
            torch.save(paths.concat(self.dirTar, 'pack_v.t7'), {unpack(self.t7Tar, self.offset + 1)})
            print('\tTrain set: ' .. self.dirTar .. '/pack.t7 loaded')

            self.t7Inp = {}
            for i = 1, #self.dirInp do
                if self.scale[i] ~= 1 then
                    table.insert(self.t7Inp, torch.load(paths.concat(self.dirInp[i], 'pack.t7')))
                    torch.save(paths.concat(self.dirInp[i], 'pack_v.t7'), {unpack(self.t7Inp[i], self.offset + 1)})
                    print('\tTrain set: ' .. self.dirInp[i] .. '/pack.t7 loaded')
                else
                    table.insert(self.t7Inp, self.t7Tar)
                end
            end
        elseif split == 'val' then
            self.t7Tar = torch.load(paths.concat(self.dirTar, 'pack_v.t7'))
            print('\tValidation set: ' .. self.dirTar .. '/pack_v.t7 loaded')
            self.t7Inp = {}
            for i = 1, #self.dirInp do
                if self.scale[i] ~= 1 then
                    table.insert(self.t7Inp, torch.load(paths.concat(self.dirInp[i], 'pack_v.t7')))
                    print('\tValidation set: ' .. self.dirInp[i] .. '/pack_v.t7 loaded')
                else
                    table.insert(self.t7Inp, self.t7Tar)
                end
            end
        end
    end

    collectgarbage()
    collectgarbage()
end

function div2k:get(idx, scaleIdx)
    local scale = self.scale[scaleIdx]
    local dataSize = self.dataSize

    if (self.split == 'val') and (self.opt.datatype ~= 't7pack') then
        idx = idx + self.offset
    end

    local input, target

    if self.opt.datatype == 't7pack' then
        input = self.t7Inp[scaleIdx][idx]
        target = self.t7Tar[idx]
    else
        local inputName, targetName = self:getFileName(idx, scale)
        if self.ext == '.png' then
            input = image.load(paths.concat(self.dirInp[scaleIdx], inputName), self.opt.nChannel, 'float')
            target = image.load(paths.concat(self.dirTar, targetName), self.opt.nChannel, 'float')
        else
            input = torch.load(paths.concat(self.dirInp[scaleIdx], inputName))
            target = torch.load(paths.concat(self.dirTar, targetName))
        end
    end

    local channel, h, w = unpack(target:size():totable())
    local hInput, wInput = math.floor(h / scale), math.floor(w / scale)
    local hTarget, wTarget = scale * hInput, scale * wInput
    if dataSize == 'big' then
        hInput, wInput = hTarget, wTarget
    end
    target = target[{{}, {1, hTarget}, {1, wTarget}}]
    local patchSize = self.opt.patchSize
    local targetPatch = patchSize
    local inputPatch = (dataSize == 'big') and patchSize or (patchSize / scale)
    if (wTarget < targetPatch) or (hTarget < targetPatch) then
        return nil
    end

    --Generate patches for training
    if self.split == 'train' then
        local ix = torch.random(1, wInput - inputPatch + 1)
        local iy = torch.random(1, hInput - inputPatch + 1)
        local tx, ty = ix, iy
        if dataSize == 'small' then
            tx, ty = (scale * (ix - 1)) + 1, (scale * (iy - 1)) + 1
        end
        input = input[{{}, {iy, iy + inputPatch - 1}, {ix, ix + inputPatch - 1}}]
        target = target[{{}, {ty, ty + targetPatch - 1}, {tx, tx + targetPatch - 1}}]
    end

    if self.ext == '.png' then
        input:mul(self.opt.mulImg)
        target:mul(self.opt.mulImg)
    else
        input = input:float():mul(self.opt.mulImg / 255)
        target = target:float():mul(self.opt.mulImg / 255)
    end

    --Reject the patch that has small size of spatial gradient
    if (self.split == 'train') and (self.opt.rejection ~= -1) then
        local ni = input / self.opt.mulImg
        local dx = image.crop(ni - image.translate(ni, -1, 0), 'tl', inputPatch - 1, inputPatch - 1)
        local dy = image.crop(ni - image.translate(ni, 0, -1), 'tl', inputPatch - 1, inputPatch - 1)
        local dsum = dx:pow(2) + dy:pow(2)
        local dsqrt = dsum:sqrt()
        local gradValue = dsqrt:view(-1):mean()
        if gradValue <= self.opt.rejection then
            return nil
        end
    end

    return {
        input = input,
        target = target
    }
end

function div2k:__size()
    if self.split == 'train' then
        return self.offset
    elseif self.split == 'val' then
        return self.opt.numVal
    end
end

function div2k:augment()
    if (self.split == 'train') and (self.opt.degrade == 'bicubic') then
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

function div2k:getFileName(idx, scale)
    --filename format: ????x?.png
    local fileName = idx
    local digit = idx
    while (digit < 1000) do
        fileName = '0' .. fileName
        digit = digit * 10
    end

    local targetName = fileName .. self.ext
    local inputName = nil
    if scale == 1 then
        inputName = targetName
    else
        if self.opt.netType ~= 'recurVDSR' then
            inputName = fileName .. 'x' .. scale .. self.ext
        else
            inputName = 'SRres' .. fileName .. 'x' .. scale .. self.ext
        end
    end

    return inputName, targetName
end

return M.div2k
