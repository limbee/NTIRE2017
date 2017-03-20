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
    local apath = paths.concat(opt.datadir, 'dataset/DIV2K')
 
    if opt.datatype ~= 't7pack' then
        self.dirTar = paths.concat(apath, 'DIV2K_train_HR')
        self.dirInp = paths.concat(apath, 'DIV2K_train_LR_' .. opt.degrade, 'X' .. opt.scale)

        if opt.dataSize == 'big' then
            self.dirInp = self.dirInp .. 'b'
        end
        if opt.netType == 'recurVDSR' then  --SRresOutput
            self.dirInp = self.dirInp .. '_SRresOutput'
        end
    else
        self.dirTar = paths.concat(apath, 'DIV2K_decoded', 'DIV2K_train_HR')
        self.dirInp = paths.concat(apath, 'DIV2K_decoded', 'DIV2K_train_LR_' .. opt.degrade .. '_X' .. opt.scale)

        if opt.dataSize == 'big' then
            self.dirInp = self.dirInp .. 'b'
        end
        if split == 'train' then
            self.t7Tar = torch.load(self.dirTar .. '.t7')
            self.t7Inp = torch.load(self.dirInp .. '.t7')
            
            if not paths.filep(self.dirTar .. 'v.t7') or not paths.filep(self.dirInp .. 'v.t7') then
                local valTar = {}
                local valInp = {}

                for i = self.size - opt.numVal + 1, self.size do
                    table.insert(valTar, self.t7Tar[i])
                    table.insert(valInp, self.t7Inp[i])
                end

                torch.save(self.dirTar .. 'v.t7', valTar)
                torch.save(self.dirInp .. 'v.t7', valInp)

                valTar = nil
                valInp = nil
                collectgarbage()
                collectgarbage()
            end
        elseif split == 'val' then
            self.t7Tar = torch.load(self.dirTar .. 'v.t7')
            self.t7Inp = torch.load(self.dirInp .. 'v.t7')
        end
    end 
 end

function div2k:get(i)
    local netType = self.opt.netType
    local dataSize = self.opt.dataSize
    local idx = i
    if (self.split == 'val') and (self.opt.datatype ~= 't7pack') then
        idx = idx + (self.size - self.opt.numVal)
    end

    local input, target
    local scale = self.opt.scale
    local ext = (self.opt.datatype == 'png') and '.png' or '.t7'
    
    if self.opt.datatype == 't7pack' then
        input = self.t7Inp[idx]
        target = self.t7Tar[idx]
    else
       local inputName, targetName = self:getFileName(idx, ext)

        if ext == '.png' then
            input = image.load(paths.concat(self.dirInp, inputName), self.opt.nChannel, 'float')
            target = image.load(paths.concat(self.dirTar, targetName), self.opt.nChannel, 'float')
        else
            input = torch.load(paths.concat(self.dirInp, inputName)):float()
            target = torch.load(paths.concat(self.dirTar, targetName)):float()
        end
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

    if ext == '.t7' then
        input = input:float():mul(self.opt.mulImg / 255)
        target = target:float():mul(self.opt.mulImg / 255)
    else
        input:mul(self.opt.mulImg)
        target:mul(self.opt.mulImg)
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
