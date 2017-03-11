require 'nn'
require 'cunn'

local function createModel(opt)

    local fltsz = opt.filtsize
    local padsz = (fltsz-1)/2
    local nBlocks = opt.nResBlock
    local nFeat = opt.nFeat
    local nChannel = opt.nChannel
    local scale = opt.scale

    local function ResBlock(fltsz, nFeat)
        local fltsz = fltsz or opt.filtsize
        local padsz = (fltsz-1)/2
        local nFeat = nFeat or opt.nFeat

        local conv1 = nn.SpatialConvolution(nFeat,nFeat, fltsz,fltsz, 1,1, padsz,padsz)
        local relu = nn.ReLU(true)
        local conv2 = nn.SpatialConvolution(nFeat,nFeat, fltsz,fltsz, 1,1, padsz,padsz)

        local path = nn.Sequential()
            :add(conv1):add(relu)
            :add(conv2)

        local concat = nn.ConcatTable()
            :add(path)
            :add(nn.Identity())

        local block = nn.Sequential()
            :add(concat)
            :add(nn.CAddTable(true))

        return block
    end

    local function ResNet(nBlocks, fltsz, nFeat)
        local model = nn.Sequential()
        local block = ResBlock(fltsz, nFeat)
        for i = 1, nBlocks do
            model:add(block:get(1):clone())
            model:add(block:get(2):clone())
        end
        block = nil

        return model
    end

    local function Upsampler(scale, inFeat, nFeat)
        local model = nn.Sequential()
        if scale == 2 or scale == 3 then
            model:add(nn.SpatialConvolution(inFeat,nFeat*scale^2, fltsz,fltsz, 1,1, padsz,padsz), 1)
            model:add(nn.PixelShuffle(scale))
        elseif scale == 4 then
            model:add(nn.SpatialConvolution(inFeat,nFeat*scale, fltsz,fltsz, 1,1, padsz,padsz), 1)
            model:add(nn.PixelShuffle(scale^0.5))
            model:add(nn.SpatialConvolution(nFeat,nFeat*scale, fltsz,fltsz, 1,1, padsz,padsz), 1)
            model:add(nn.PixelShuffle(scale^0.5))
        else
            error('Unknown scale')
        end

        return model
    end

    local model = nn.Sequential()
        :add(nn.SpatialConvolution(nChannel,nFeat, fltsz,fltsz, 1,1, padsz,padsz))
        
    local concat = nn.Concat(2)
    do
        local path1 = ResNet(nBlocks, fltsz, nFeat)
        local upsampler = Upsampler(scale, nFeat, nFeat)
        for i = 1, upsampler:size() do
            path1:insert(upsampler:get(i), i)
        end
        concat:add(path1)
    end
    do
        local path2 = ResNet(nBlocks, fltsz, nFeat)
        local upsampler = Upsampler(scale, nFeat, nFeat)
        for i = 1, upsampler:size() do
            path2:insert(upsampler:get(i), i)
        end
        concat:add(path2)
    end
    model:add(concat)
    model:add(nn.SpatialConvolution(nFeat*2, nChannel, fltsz,fltsz,1,1,padsz,padsz))
    model:reset()

    return model
end

return createModel