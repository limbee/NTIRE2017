local image = require 'image'
local paths = require 'paths'
local transform = require 'data/transforms'
local ffi = require 'ffi'
local util = require 'utils'()
--local cv = require 'cv'
--require 'cv.imgcodecs'
--require 'cv.imgproc'

local M = {}
local dataset = torch.class('sr.dataset', M)

function dataset:__init(opt, split)
    self.imgPaths = {}
    self.area = {}
    if split=='train' then
        print('\tloading train data...')
        if opt.dataset=='imagenet' then
            local f
            if opt.trainset == 'train' then
                self.dirTar = '../dataset/ILSVRC2015/Data/CLS-LOC/train/'
                f = io.open('../dataset/ILSVRC2015/ImageSets/CLS-LOC/train_loc.txt')
            elseif opt.trainset == 'val' then
                self.dirTar = '../dataset/ILSVRC2015/Data/CLS-LOC/val/'
                f = io.open('../dataset/ILSVRC2015/ImageSets/CLS-LOC/val.txt')
            else
                error('wrong option: trainset')
            end
            while true do
                local line = f:read('*l')
                if not line then break end
                local imgPath = line:split(' ')[1] ..'.JPEG'
                self.imgPaths[#self.imgPaths+1] = imgPath
            end
        elseif opt.dataset=='91' or opt.dataset=='291' then
            self.imgTar = {}
            self.imgInp = {}
            if opt.netType=='VDSR' then for sc=2,4 do self.imgInp[sc] = {} end end
            self.dirTar = paths.concat('../dataset',opt.dataset)
            for file in paths.iterfiles(self.dirTar) do
                local img = image.load(paths.concat(self.dirTar,file))
                local area
                if img:dim()==2 then
                    area = img:size(1) * img:size(2)
                elseif img:dim()==3 then
                    area = img:size(2) * img:size(3)
                else
                    goto skip
                end
                self.imgPaths[#self.imgPaths+1] = file
                self.area[#self.area+1] = area
                self.imgTar[#self.imgTar+1] = img
                if opt.netType=='VDSR' then
                    for sc=2,4 do
                        local inp = image.load(paths.concat(self.dirTar,'big','x'..sc,file:split('%.')[1] .. '.png'))
                        self.imgInp[sc][#self.imgInp[sc]+1] = inp
                    end
                else
                    local inp = image.load(paths.concat(self.dirTar,'small',file:split('%.')[1] .. '.png'))
                    self.imgInp[#self.imgInp+1] = inp
                end
                ::skip::
            end
            self.area = torch.Tensor(self.area)
        end
        self.dirInp = opt.netType=='VDSR' and paths.concat(self.dirTar,'big') or paths.concat(self.dirTar,'small')
        --[[
        --local cachePath = paths.concat('../dataset/cachePath_train.t7')
        --local cachePath = paths.concat('../dataset/cachePath_val.t7')
        if paths.filep(cachePath) then
            print('loading train data list from cache...')
            self.imgPaths = torch.load(cachePath)
        else
        --]]
            --torch.save(cachePath,self.imgPaths)
            --print('Saved trainset cache')
        --end
    elseif split=='val' then
        print('\tloading validation data...')
        if opt.valset == 'val' then
            local numVal = 10
            self.dirTar = '../dataset/ILSVRC2015/Data/CLS-LOC/val/'
            self.dirInp = opt.netType=='VDSR' and paths.concat(self.dirTar,'big') or paths.concat(self.dirTar,'small')
            local rand = torch.randperm(50002)[{{1,numVal+2}}]
            self.imgPaths = {}
            local tmp = paths.dir(self.dirTar)
            for i = 1,rand:size(1) do
                if tmp[rand[i]] ~= '.' and tmp[rand[i]] ~= '..' then
                    self.imgPaths[i] = tmp[rand[i]]
                end
                if #self.imgPaths == numVal then break end
            end
        else
            self.dirTar = paths.concat('../dataset/benchmark',opt.valset)
            self.dirInp = opt.netType=='VDSR' 
                    and paths.concat('../dataset/benchmark','big',opt.valset)
                    or paths.concat('../dataset/benchmark','small',opt.valset)

            for file in paths.iterfiles(self.dirTar) do
                self.imgPaths[#self.imgPaths+1] = file
            end
        end
    end

    self.opt = opt
    self.split = split
end

function dataset:get(i)
--[[
    local img = cv.imread{paths.concat(self.dirTar,self.imgPaths[i]), cv.IMREAD_COLOR}
    img = img:permute(3,1,2):index(1,torch.LongTensor{3,2,1}):float()/255
    local _,h,w = table.unpack(img:size():totable())
    local inter
    if self.opt.inter == 'bicubic' then inter = cv.bicubic
    elseif self.opt.inter == 'inter_area' then inter = cv.inter_area end
    local hh,ww = self.opt.scale*math.floor(h/self.opt.scale), self.opt.scale*math.floor(w/self.opt.scale)
    local target = img[{{},{1,hh},{1,ww}}]
    local hhi,wwi = hh/self.opt.scale, ww/self.opt.scale
    local target_blur = cv.GaussianBlur{src=target, ksize={3,3}, sigmaX = self.opt.sigma}
    local input = cv.resize{target_blur, {wwi,hhi}, interpolation=inter}
--]]
    local function get_img(scale)
        local name= math.random(1,1e5)

        local target
        if self.split=='train' and (self.opt.dataset=='91' or self.opt.dataset=='291') then
            target = self.imgTar[i]:clone()
        else
            target = image.load(paths.concat(self.dirTar,self.imgPaths[i]))
        end
        if target:dim()==2 or (target:dim()==3 and target:size(1)==1) then target = target:repeatTensor(3,1,1) end
        if target:size(1)~=3 then return end
        local _,h,w = table.unpack(target:size():totable())
        local hh,ww = scale*math.floor(h/scale), scale*math.floor(w/scale)
        target = target[{{},{1,hh},{1,ww}}]
        local input

        if self.opt.netType=='VDSR' then
            inputName = paths.concat(self.dirInp,'x' .. scale,self.imgPaths[i]):gsub('%.%w+','.png')
            if self.split=='train' and (self.opt.dataset=='91' or self.opt.dataset=='291') then
                if self.opt.netType=='VDSR' then
                    input = self.imgInp[scale][i]:clone()
                else 
                    input = self.imgInp[i]:clone()
                end
            else
                input = image.load(inputName)
            end
            
            if self.split == 'train' then
                local ps = self.opt.patchSize
                if ww < ps or hh < ps then return end

                local x = torch.random(1,ww-ps+1)
                local y = torch.random(1,hh-ps+1)

                input = input[{{},{y,y+ps-1},{x,x+ps-1}}]
                target = target[{{},{y,y+ps-1},{x,x+ps-1}}]
            end
        else
            local inputName = paths.concat(self.dirInp,self.imgPaths[i]):gsub('%.%w+','.png')
            input = image.load(inputName)
            local hhi,wwi = hh/scale, ww/scale

            if self.split == 'train' then 
                local tps = self.opt.patchSize -- target patch size
                local ips = self.opt.patchSize / scale -- input patch size
                if ww < tps or hh < tps then return end

                local ix = torch.random(1, wwi-ips+1)
                local iy = torch.random(1, hhi-ips+1)
                local tx = scale*(ix-1)+1
                local ty = scale*(iy-1)+1

                input = input[{{},{iy,iy+ips-1},{ix,ix+ips-1}}]
                target = target[{{},{ty,ty+tps-1},{tx,tx+tps-1}}]
            end
        end

        input:mul(255)
        target:mul(255)

        if self.opt.nChannel == 1 then
            input = util:rgb2y(input)
            target = util:rgb2y(target)
        end
        return {
            input = input,
            target = target
        }
    end

    if self.opt.netType=='VDSR' then
        if self.split=='val' then
            local input,target = {},{}
            for scale=2,4 do
                local sample = get_img(scale)
                input[#input+1] = sample.input
                target[#target+1] = sample.target
            end
            return {
                input = input,
                target = target
            }
        else
            return get_img(math.random(2,4))
        end
    else
        return get_img(self.opt.scale)
    end
end

function dataset:size()
    return #self.imgPaths
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

function dataset:augment()
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

return M.dataset
