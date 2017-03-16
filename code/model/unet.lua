require 'nn'
require 'cunn'

----------------------------------------------------------------------------
-- This file doesn't make the original u-net (Ronneberger 2015)
-- But the overall structure using the skip connection imitated u-net.
-- We don't use any pooling operation here.
----------------------------------------------------------------------------

local function createModel(opt)
    local nFeat = opt.nFeat
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization
    local shuffle = nn.PixelShuffle
    local pad = nn.Padding
    local seq = nn.Sequential
    local concat = nn.ConcatTable
    local id = nn.Identity
    local cadd = nn.CAddTable
    local deconv = nn.SpatialFullConvolution

    local function cbrcb(nFeat)
        return seq()
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
            :add(relu(true))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
    end

    local function brcbrc(nFeat)
        return seq()
            :add(bnorm(nFeat))
            :add(relu(true))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
            :add(relu(true))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
    end

    local function crc(nFeat)
        return seq()
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(relu(true))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
    end

    local function resBlock(nFeat)
        return seq()
            :add(concat()
                :add(cbrcb(nFeat))
                :add(id()))
            :add(cadd(true))
    end

    local function resConvBN(nFeat)
        return seq()
            :add(resBlock)
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
    end

    local function addSkip(layers)
        return seq()
            :add(concat()
                :add(layers)
                :add(id()))
            :add(cadd(true))
    end


    local body

    if opt.modelVer == 1 then
        body = resBlock(nFeat)
        for i = 1, (opt.nConv - 4) / 4 do
            body = addSkip(seq()
                :add(cbrcb(nFeat))
                :add(body)
                :add(cbrcb(nFeat)))
        end
    elseif opt.modelVer == 2 then
        body = addSkip(seq()
            :add(cbrcb(nFeat))
            :add(cbrcb(nFeat)))
        for i = 1, (opt.nConv - 6) / 8 do
            body = addSkip(seq()
                :add(cbrcb(nFeat))
                :add(cbrcb(nFeat))
                :add(body)
                :add(cbrcb(nFeat))
                :add(cbrcb(nFeat)))
        end
    elseif opt.modelVer == 3 then
        body = resBlock(nFeat)
        for i = 1, (opt.nConv - 4) / 4 do
            body = addSkip(seq()
                :add(resBlock(nFeat))
                :add(body)
                :add(resBlock(nFeat)))
        end
    
    -------------------------------------------------
    -- Version 1 slightly wins the 2 and 3.
    -- However, none of 1 ~ 3 succeeded in beating the resnet.
    -------------------------------------------------

    elseif opt.modelVer == 4 then
        body = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 5) / 2 do
            body = addSkip(seq()
                :add(cbrcb(nFeat))
                :add(body))
        end
    elseif opt.modelVer == 5 then
        bocy = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 5) / 2 do
            body = addSkip(seq()
                :add(resBlock(nFeat))
                :add(body))
        end 
    elseif opt.modelVer == 6 then
        body = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 5) / 4 do
            body = addSkip(seq()
                :add(cbrcb(nFeat))
                :add(relu(true))
                :add(cbrcb(nFeat))
                :add(relu(true))
                :add(body))
        end 
    elseif opt.modelVer == 7 then
        body = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 4) / 4 do
            body = addSkip(seq()
                :add(resBlock(nFeat))
                :add(resBlock(nFeat))
                :add(body))
        end
    elseif opt.modelVer == 8 then
        body = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 4) / 8 do
            body = addSkip(seq()
                :add(resBlock(nFeat))
                :add(resBlock(nFeat))
                :add(resBlock(nFeat))
                :add(resBlock(nFeat))
                :add(body))
        end
    else

    end



    model = seq()
        :add(conv(opt.nChannel,nFeat, 3,3, 1,1, 1,1))
        :add(relu(true))
        :add(body)
        :add(require 'model/upsample'(opt))
        :add(conv(nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    return model
end

return createModel