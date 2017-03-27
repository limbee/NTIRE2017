
local paths = require 'paths'
local transform = require 'data/transforms'

local M = {}
local div2k = torch.class('sr.div2k', M)

function div2k:__init(opt, split)
    self.size = 800
    self.opt = opt
    self.split = split
    self.scale = opt.scale

    --absolute path of the dataset
    local apath = paths.concat(opt.datadir, 'dataset/DIV2K')
    local tHR = 'DIV2K_train_HR'
    local tLR = 'DIV2K_train_LR_'
    local dec = 'DIV2K_decoded'

    if opt.datatype ~= 't7pack' then
        self.dirTar = paths.concat(apath, tHR)
        self.dirInp = paths.concat(apath, tLR .. opt.degrade, 'X' .. opt.scale)

        if opt.dataSize == 'big' then
            self.dirInp = self.dirInp .. 'b'
        end
        if opt.netType == 'recurVDSR' then  --SRresOutput
            self.dirInp = self.dirInp .. '_SRresOutput'
        end
    else
        self.dirTar = paths.concat(apath, dec, tHR)
        self.dirInp = {}
        for i = 1, #self.scale do
            table.insert(self.dirInp,
                    paths.concat(apath, dec, tLR .. opt.degrade .. '_X' .. self.scale[i]))
            if opt.dataSize == 'big' then
                self.dirInp[i] = self.dirInp[i] .. 'b'
            end
        end
        if split == 'train' then
            self.t7Tar = torch.load(self.dirTar .. '.t7')
            if not paths.filep(self.dirTar .. 'v.t7') then
                local valTar = {}
                for i = (self.size - opt.numVal + 1), self.size do
                    table.insert(valTar, self.t7Tar[i])
                end
                torch.save(self.dirTar .. 'v.t7', valTar)
                valTar = nil
                collectgarbage()
                collectgarbage()
            end

            self.t7Inp = {}
            for i = 1, #self.dirInp do
                table.insert(self.t7Inp, torch.load(self.dirInp[i] .. '.t7'))
                if not paths.filep(self.dirInp[i] .. 'v.t7') then
                    local valInp = {}
                    for j = (self.size - opt.numVal + 1), self.size do
                        table.insert(valInp, self.t7Inp[i][j])
                    end
                    torch.save(self.dirInp[i] .. 'v.t7', valInp)
                    valInp = nil
                    collectgarbage()
                    collectgarbage()
                end
            end
        elseif split == 'val' then
            self.t7Tar = torch.load(self.dirTar .. 'v.t7')
            self.t7Inp = {}
            for i = 1, #self.dirInp do
                table.insert(self.t7Inp, torch.load(self.dirInp[i] .. 'v.t7'))
            end
        end
    end 
end

function div2k:get(i, scaleR)
    local scale = self.scale[scaleR]
    local netType = self.opt.netType
    local dataSize = self.opt.dataSize
    local idx = i
    if (self.split == 'val') and (self.opt.datatype ~= 't7pack') then
        idx = idx + (self.size - self.opt.numVal)
    end

    local input, target
    local ext = (self.opt.datatype == 'png') and '.png' or '.t7'

    if self.opt.datatype == 't7pack' then
        input = self.t7Inp[scaleR][idx]
        target = self.t7Tar[idx]
    else
        local inputName, targetName = self:getFileName(idx, ext)

        if ext == '.png' then
            input = image.load(paths.concat(self.dirInp, inputName), self.opt.nChannel, 'float')
            target = image.load(paths.concat(self.dirTar, targetName), self.opt.nChannel, 'float')
        else
            input = torch.load(paths.concat(self.dirInp, inputName)):float()
        end
    end

    local channel, h, w = table.unpack(target:size():totable())
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

    if self.split == 'train' then 
        local ix = torch.random(1, wInput - inputPatch + 1)
        local iy = torch.random(1, hInput - inputPatch + 1)
        local tx, ty = ix, iy
        if dataSize ~= 'big' then
            tx, ty = (scale * (ix - 1)) + 1, (scale * (iy - 1)) + 1
        end
        
        input = input[{{}, {iy, iy + inputPatch - 1}, {ix, ix + inputPatch - 1}}]
        target = target[{{}, {ty, ty + targetPatch - 1}, {tx, tx + targetPatch - 1}}]
    end

    if ext == '.t7' then
        input = input:float():mul(self.opt.mulImg / 255)
        target = target:float():mul(self.opt.mulImg / 255)
    else
        input:mul(self.opt.mulImg)
        target:mul(self.opt.mulImg)
    end
    
    --reject the patch that has small size of spatial gradient
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
        -- We don't need vertical flip, since hflip + rotation covers it
        table.insert(transforms, transform.HorizontalFlip())
        table.insert(transforms, transform.Rotation())

        return transform.Compose(transforms)
    elseif self.split == 'val' then
        return function(sample) return sample end
    end
end

function div2k:getFileName(idx, ext)
    --filename format: ????x?.png
    local fileName = idx
    local digit = idx
    while (digit < 1000) do
        fileName = '0' .. fileName
        digit = digit * 10
    end

    local inputName
    if self.opt.netType == 'recurVDSR' then
        inputName = 'SRres' .. fileName .. 'x' .. self.opt.scale .. ext
    else
        inputName = fileName .. 'x' .. self.opt.scale .. ext
    end
    local targetName = fileName .. ext

    return inputName, targetName
end

return M.div2k
