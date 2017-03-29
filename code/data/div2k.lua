
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

    if self.datatype == 'png' then
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
                table.insert(self.t7Inp, torch.load(paths.concat(self.dirInp[i], 'pack.t7')))
                torch.save(paths.concat(self.dirInp[i], 'pack_v.t7'), {unpack(self.t7Inp[i], self.offset + 1)})
                print('\tTrain set: ' .. self.dirInp[i] .. '/pack.t7 loaded')
            end
        elseif split == 'val' then
            self.t7Tar = torch.load(paths.concat(self.dirTar, 'pack_v.t7'))
            print('\tValidation set: ' .. self.dirTar .. '/pack_v.t7 loaded')
            self.t7Inp = {}
            for i = 1, #self.dirInp do
                table.insert(self.t7Inp, torch.load(paths.concat(self.dirInp[i], 'pack_v.t7')))
                print('\tValidation set: ' .. self.dirInp[i] .. '/pack_v.t7 loaded')
            end
        end
    end

    collectgarbage()
    collectgarbage()
end

function div2k:get(idx, scaleR)
    local scale = self.scale[scaleR]
    local dataSize = self.dataSize

    if (self.split == 'val') and (self.opt.datatype ~= 't7pack') then
        idx = idx + self.offset
    end

    local _input, _target

    if self.opt.datatype == 't7pack' then
        _input = self.t7Inp[scaleR][idx]
        _target = self.t7Tar[idx]
    else
        local inputName, targetName = self:getFileName(idx, scaleR + 1)
        if self.ext == '.png' then
            _input = image.load(paths.concat(self.dirInp[scaleR], inputName), self.opt.nChannel, 'float')
            _target = image.load(paths.concat(self.dirTar, targetName), self.opt.nChannel, 'float')
        else
            _input = torch.load(paths.concat(self.dirInp[scaleR], inputName))
            _target = torch.load(paths.concat(self.dirTar, targetName))
        end
    end

    local channel, h, w = unpack(_target:size():totable())
    local hInput, wInput = math.floor(h / scale), math.floor(w / scale)
    local hTarget, wTarget = scale * hInput, scale * wInput
    if dataSize == 'big' then
        hInput, wInput = hTarget, wTarget
    end
    _target = _target[{{}, {1, hTarget}, {1, wTarget}}]
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
        _input = _input[{{}, {iy, iy + inputPatch - 1}, {ix, ix + inputPatch - 1}}]
        _target = _target[{{}, {ty, ty + targetPatch - 1}, {tx, tx + targetPatch - 1}}]
    end

    if self.ext == '.png' then
        _input:mul(self.opt.mulImg)
        _target:mul(self.opt.mulImg)
    else
        _input = _input:float():mul(self.opt.mulImg / 255)
        _target = _target:float():mul(self.opt.mulImg / 255)
    end

    --Reject the patch that has small size of spatial gradient
    if (self.split == 'train') and (self.opt.rejection ~= -1) then
        local ni = _input / self.opt.mulImg
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
        input = _input,
        target = _target
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
        -- We don't need vertical flip, since hflip + rotation covers it
        table.insert(transforms, transform.HorizontalFlip())
        table.insert(transforms, transform.Rotation())

        return transform.Compose(transforms)
    elseif self.split == 'val' then
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

    local inputName = nil
    if self.opt.netType == 'recurVDSR' then
        inputName = 'SRres' .. fileName .. 'x' .. scale .. self.ext
    else
        inputName = fileName .. 'x' .. scale .. self.ext
    end
    local targetName = fileName .. self.ext

    return inputName, targetName
end

return M.div2k
