require 'nn'
require 'cunn'

local function createModel(opt)
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization
    local shuffle = nn.PixelShuffle
    local pad = nn.Padding

    local function resBlock(nFeat, stride, preActivation, bottleNeck)
        local seq = nn.Sequential()
        if (bottleNeck) then
            if (not preActivation) then
                seq:add(conv(nFeat, nFeat, 1, 1))
                seq:add(bnorm(nFeat))
                seq:add(relu(true))
            else
                seq:add(bnorm(nFeat))
                seq:add(relu(true))
                seq:add(conv(nFeat, nFeat, 1, 1))
            end
        end
        if (not preActivation) then
            seq:add(conv(nFeat, nFeat, 3, 3, stride, stride, 1, 1))
            seq:add(bnorm(nFeat))
            seq:add(relu(true))
            seq:add(conv(nFeat, nFeat, 3, 3, 1, 1, 1, 1))
            seq:add(bnorm(nFeat))
        else
            seq:add(bnorm(nFeat))
            seq:add(relu(true))
            seq:add(conv(nFeat, nFeat, 3, 3, stride, stride, 1, 1))
            seq:add(bnorm(nFeat))
            seq:add(relu(true))
            seq:add(conv(nFeat, nFeat, 3, 3, 1, 1, 1, 1))
        end
        return nn.Sequential()
        :add(nn.ConcatTable()
            :add(seq)
            :add(nn.Identity())
        )
        :add(nn.CAddTable(true))
    end
    
    local preActivation = opt.pre_act
    local blockType = opt.bottleNeck

    local model = nn.Sequential()
    local cat = nn.ConcatTable()
    local resNet = require('model/resnet')(opt)
    
    local highNet = nn.Sequential()
    highNet:add(conv(opt.nChannel, opt.nFeat, 3, 3, 1, 1, 1, 1))
    highNet:add(relu(true))

    for i = 1, opt.nResBlock do
        highNet:add(resBlock(opt.nFeat, 1, preActivation, blockType))
    end
    highNet:add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
    highNet:add(bnorm(opt.nFeat))

    if (opt.upsample == 'full') then
        highNet:add(nn.SpatialFullConvolution(opt.nFeat, opt.nFeat, 4, 4, 2, 2, 1, 1))
        highNet:add(relu(true))
    elseif (opt.upsample == 'shuffle') then
        highNet:add(pad(2, 1, 3))
        highNet:add(pad(3, 1, 3))
        highNet:add(conv(opt.nFeat, opt.nFeat * 4, 4, 4, 1, 1, 1, 1))
        highNet:add(shuffle(2))
        highNet:add(relu(true))
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
