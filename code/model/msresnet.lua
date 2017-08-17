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
    local shuffle = nn.PixelShuffle
    local seq = nn.Sequential
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

    local model = seq()
        :add(bias(-opt.mulImg/2, true))
        :add(conv(nChannel,nFeat, fltsz,fltsz, 1,1, padsz,padsz))
    
    assert(nBlocks/2 == math.floor(nBlocks/2), 'nBlocks should be an even number')
    if opt.modelVer == 1 then
        local trunk = concat()
        do  -- fine
            local path1 = ResNet(nBlocks/2, nFeat, fltsz)
            local upsampler = Upsampler(scale, nFeat, nFeat)
            for i = 1, upsampler:size() do
                path1:insert(upsampler:get(i):clone(), i)
            end
            trunk:add(path1)
        end
        do  -- mid
            local path2 = ResNet(nBlocks, nFeat, fltsz)
            local upsampler = Upsampler(scale, nFeat, nFeat)
            for i = 1, upsampler:size() do
                path2:add(upsampler:get(i):clone())
            end
            trunk:add(path2)
        end
        
        model:add(trunk)
        model:add(join(2))

        model:add(conv(nFeat*2,nChannel, fltsz,fltsz, 1,1, padsz,padsz))
    elseif opt.modelVer == 2 then
        do	-- front
            local path1 = Upsampler(scale, nFeat, nFeat)
            local path2 = ResNet(nBlocks, nFeat, fltsz)
            local upsampler = Upsampler(scale, nFeat, nFeat)
            for i = 1, upsampler:size() do
                path2:add(upsampler:get(i):clone())
            end

            local trunk = concat()
                :add(path1)
                :add(path2)
            
            model:add(trunk)
            model:add(join(2))
        end
        do	-- back
            model:add(conv(nFeat*2,nFeat, fltsz,fltsz, 1,1, padsz,padsz))
            local path3 = ResNet(nBlocks/2, nFeat, fltsz)
            for i = 1, path3:size() do
                model:add(path3:get(i):clone())
            end
        end
        model:add(conv(nFeat,nChannel, fltsz,fltsz, 1,1, padsz,padsz))
    else
        error('Unknown version of MSResNet!')
    end
    model:add(bias(opt.mulImg/2, true))
    model:reset()

    collectgarbage()
    collectgarbage()
    
    return model
end

return createModel
