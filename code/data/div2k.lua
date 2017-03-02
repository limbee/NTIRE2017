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

    --target dataset, input dataset
    self.dataTarget = nil
    self.dataInput = nil
    self.dirTar = nil
    self.dirInp = nil
    
    --absolute path of the dataset
    local apath = '/var/tmp/dataset/DIV2K'
    if (opt.datatype == 'png') then
        self.dirTar = apath .. '/DIV2K_train_HR'
        self.dirInp = paths.concat(apath .. '/DIV2K_train_LR_' .. opt.degrade, 'X' .. opt.scale)
    elseif (opt.datatype == 't7') then
        local dirDecoded = apath .. '/DIV2K_decoded'
        self.dirTar = dirDecoded .. '/DIV2K_train_HR.t7'
        if (opt.netType == 'vdsr') then
            self.dirInp = dirDecoded .. '/DIV2k_train_LR_' .. opt.degrade .. '_X' .. opt.scale .. 'b.t7'
        else
            self.dirInp = dirDecoded .. '/DIV2k_train_LR_' .. opt.degrade .. '_X' .. opt.scale .. '.t7'
        end
        self.dataTarget = torch.load(dirTar)
        self.dataInput = torch.load(dirInp)
    end
end

function div2k:get(i)
    local idx = nil
    if (self.split == 'train') then
        idx = i
    elseif (self.split == 'val') then
        idx = (self.size - self.opt.numVal + i)
    end

    local scale = self.opt.scale
    local target = nil
    local input = nil

    if (self.opt.datatype == 'png') then
        local tarName = ''
        local nDigit = math.floor(math.log10(i)) + 1
        for j = 1, (4 - nDigit) do
            tarName = tarName .. '0'
        end
        tarName = tarName .. i
        inpName = tarName .. 'x' .. scale .. '.png'
        tarName = tarName .. '.png'

        target = image.load(paths.concat(self.dirTar, tarName), self.opt.nChannel, 'float')
        input = image.load(paths.concat(self.dirInp, inpName), self.opt.nChannel, 'float')
    elseif (self.opt.datatype == 't7') then
        target = self.dataTarget[i]:float()
        input = self.dataInput[i]:float()
    end

    local ch, h, w = table.unpack(target:size():totable())
    local hh, ww = (scale * math.floor(h / scale)), (scale * math.floor(w / scale))
    local hhi, wwi = (hh / scale), (ww / scale)
    target = target[{{}, {1, hh}, {1, ww}}]

    if (self.split == 'train') then 
        local tps = self.opt.patchSize -- target patch size
        local ips = self.opt.patchSize / scale -- input patch size
        if ((ww < tps) or (hh < tps)) then
            return
        end

        local ix = torch.random(1, wwi - ips + 1)
        local iy = torch.random(1, hhi - ips + 1)
        local tx = (scale * (ix - 1)) + 1
        local ty = (scale * (iy - 1)) + 1

        input = input[{{}, {iy, iy + ips - 1}, {ix, ix + ips - 1}}]
        target = target[{{}, {ty , ty + tps - 1}, {tx, tx + tps - 1}}]
    end

    if (self.opt.datatype == 'png') then
        input:mul(255)
        target:mul(255)
    end

    if (self.opt.nChannel == 1) then
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
