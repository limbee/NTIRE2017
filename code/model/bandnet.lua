require 'nn'
require 'cunn'

local function createModel(opt)
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization
    local shuffle = nn.PixelShuffle

    local function resBlock(nFeat, stride)
        local seq = nn.Sequential()
        seq:add(conv(nFeat, nFeat, 3, 3, stride, stride, 1, 1))
        seq:add(bnorm(nFeat))
        seq:add(relu(true))
        seq:add(conv(nFeat, nFeat, 3, 3, 1, 1, 1, 1))
        seq:add(bnorm(nFeat))

        return nn.Sequential()
        :add(nn.ConcatTable()
            :add(seq)
            :add(nn.Identity())
        )
        :add(nn.CAddTable(true))
    end

    local model = nn.Sequential()
    local cat = nn.ConcatTable()
    local resNet = require('model/resnet')(opt)
    
    local highNet = nn.Sequential()
    highNet:add(conv(opt.nChannel, opt.nFeat, 3, 3, 1, 1, 1, 1))
    highNet:add(relu(true))

    for i = 1, opt.nResBlock do
        highNet:add(resBlock(opt.nFeat, 1))
    end
    highNet:add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
    highNet:add(bnorm(opt.nFeat))

    if opt.upsample == 'full' then
        if opt.scale == 2 then
            highNet:add(nn.SpatialFullConvolution(opt.nFeat, opt.nFeat, 4, 4, 2, 2, 1, 1))
            highNet:add(relu(true))
        elseif opt.scale == 3 then
            highNet:add(nn.SpatialFullConvolution(opt.nFeat, opt.nFeat, 6, 6, 3, 3, 2, 2, 1, 1))
            highNet:add(relu(true))
        elseif opt.scale == 4 then
            highNet:add(nn.SpatialFullConvolution(opt.nFeat, opt.nFeat, 8, 8, 4, 4, 2, 2))
            highNet:add(relu(true))
        end
    elseif opt.upsample == 'shuffle' then
        if opt.scale == 2 then
            highNet:add(conv(opt.nFeat, 4 * opt.nFeat, 3, 3, 1, 1, 1, 1))
            highNet:add(shuffle(2))
            highNet:add(relu(true))
        elseif opt.scale == 3 then
            highNet:add(conv(opt.nFeat, 9 * opt.nFeat, 3, 3, 1, 1, 1, 1))
            highNet:add(shuffle(3))
            highNet:add(relu(true))
        elseif opt.scale == 4 then
            highNet:add(conv(opt.nFeat, 4 * opt.nFeat, 3, 3, 1, 1, 1, 1))
            highNet:add(shuffle(2))
            highNet:add(relu(true))
            highNet:add(conv(opt.nFeat, 4 * opt.nFeat, 3, 3, 1, 1, 1, 1))
            highNet:add(shuffle(2))
            highNet:add(relu(true))
        end
    end
    highNet:add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1))

    cat:add(resNet)
    cat:add(highNet)
    model:add(cat)
    
    model
    :add(nn.ConcatTable()
        :add(nn.Identity())
        :add(nn.CAddTable())
    )

    return model
end

return createModel
