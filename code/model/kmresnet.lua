require 'nn'
require 'cunn'

local function createModel(opt)

    local fltsz = opt.filtsize
    local padsz = (fltsz-1)/2
    local nBlocks = opt.nResBlock
    local nFeat = opt.nFeat
    local nChannel = opt.nChannel
    local scale = opt.scale[1]
    
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local seq = nn.Sequential
    local par = nn.ParallelTable
    local concat = nn.ConcatTable
    local id = nn.Identity
    local cadd = nn.CAddTable
    local bias = nn.AddConstant
    local join = nn.JoinTable
    
    local function convBlock(nFeat, fltsz)
        local padsz = (fltsz-1)/2
        local s = nn.Sequential()
            :add(conv(nFeat,nFeat, fltsz,fltsz, 1,1, padsz,padsz))
            :add(relu(true))
            :add(conv(nFeat,nFeat, fltsz,fltsz, 1,1, padsz,padsz))
        return seq()
            :add(concat()
                :add(s)
                :add(id()))
            :add(cadd(true))
    end

    local function ResNet(nBlocks, nFeat, fltsz)
        local model = seq()
        local block = convBlock(nFeat, fltsz)
        for i = 1, nBlocks do
            for j = 1, block:size() do
                model:add(block:get(j):clone())
            end
        end
        block = nil

        return model
    end

    local function Upsampler(scale, inFeat, nFeat)
        local model = seq()
        if scale == 2 or scale == 3 then
            model:add(conv(inFeat,nFeat*scale^2, fltsz,fltsz, 1,1, padsz,padsz))
            model:add(shuffle(scale))
        elseif scale == 4 then
            model:add(conv(inFeat,nFeat*scale, fltsz,fltsz, 1,1, padsz,padsz))
            model:add(shuffle(scale^0.5))
            model:add(conv(nFeat,nFeat*scale, fltsz,fltsz, 1,1, padsz,padsz))
            model:add(shuffle(scale^0.5))
        else
            error('Unknown scale')
        end

        return model
    end

    
    local model
    if opt.modelVer == 1 then
        -- Version 1. KyoungMu's ResNet
        -- input: ILR (bicubic interpolated LR)
        -- target: HR
        assert(opt.dataSize == 'big', "KyoungMu's SRResNet requires bicubic interpolation!")
        model = ResNet(nBlocks, nFeat, fltsz)
        model:insert(bias(-opt.mulImg/2, true), 1)
        model:insert(conv(nChannel,nFeat, fltsz,fltsz, 1,1, padsz,padsz), 2)

        model:add(conv(nFeat,nChannel, fltsz,fltsz, 1,1, padsz,padsz))
        model:add(bias(opt.mulImg/2, true))

    elseif opt.modelVer == 2 then
        -- Version 2. KyoungMu's Multi-scale ResNet
        -- input: {ILR, ILR_downsampled}
        -- target: {HR, LR}

        -- assert(opt.selOut == 1, 'Select fine scale output')
        -- I wonder how this variable should be used
        -- assert(opt.dataSize == 'big', "KyoungMu's SRResNet requires bicubic interpolation!")
        -- requires both HR and LR images to train this model
        
        model = seq()
        do  -- front
            local path1 = bias(-opt.mulImg/2, true)
            local path2 = ResNet(nBlocks, nFeat, fltsz)
                path2:insert(bias(-opt.mulImg/2, true), 1)
                path2:insert(conv(nChannel,nFeat, fltsz,fltsz, 1,1, padsz,padsz), 2)
                path2:add(
                    concat()
                        :add(Upsampler(scale, nFeat, nFeat-nChannel))
                        :add(id())
                )
            local front = par()
                :add(path1)
                :add(path2)
            
            model:add(front)
            model:add(nn.FlattenTable())
        end
        do  -- back
            assert(nBlocks/2 == math.floor(nBlocks/2), 'nBlocks should be an even number')
            local path1 = ResNet(nBlocks/2, nFeat, fltsz)
                path1:insert(nn.NarrowTable(1, 2), 1)
                path1:insert(nn.JoinTable(2), 2)
                path1:add(conv(nFeat,nChannel, fltsz,fltsz, 1,1, padsz,padsz))
                path1:add(bias(opt.mulImg/2, true))
            local path2 = seq()
                :add(nn.SelectTable(3))
                :add(bias(opt.mulImg/2, true))
            
            local back = concat()
                :add(path1)
                :add(path2)
            
            model:add(back)
        end

    else
        error('Unknown version of KResNet!')
    end

	model:reset()
    
    return model
end

return createModel
