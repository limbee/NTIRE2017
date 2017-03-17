require 'nn'
require 'cunn'

local function createModel(opt)
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

    if opt.modelVer == 1 then
        print('ResNet version 1!')
        print('Original ResNet')
    elseif opt.modelVer == 2 then
        print('ResNet version 2!')
        print('ResNet w/o Batch Normalization')
    elseif opt.modelVer == 3 then
        print('ResNet version 3!')
        print('ResNet w/o Batch Normalization')
        print('ResNet w/o ReLU at front & upsampling')
    else
        error("Unknown ResNet version!")
    end

    local addbn = opt.modelVer == 1
    local addrelu = (opt.modelVer == 1) or (opt.modelVer == 2)

    local function resBlock(nFeat)
        local s
        if addbn then
            s = seq()
                :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
                :add(bnorm(nFeat))
                :add(relu(true))
                :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
                :add(bnorm(nFeat))
        else
            s = seq()
                :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
                :add(relu(true))
                :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        end

        return seq()
                :add(concat()
                    :add(s)
                    :add(id()))
                :add(cadd(true))
    end

    local body = seq()
    for i=1,opt.nResBlock do
        body:add(resBlock(opt.nFeat))
    end
    body:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))
    if addbn then
        body:add(bnorm(opt.nFeat))
    end

    local model = seq()
        :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
        :add(relu(true))
        :add(concat()
            :add(body)
            :add(id()))
        :add(cadd(true))
    if not addrelu then
        model:remove(2)
    end

    local upsampler = require 'model/upsample'(opt)
    if not addrelu then
        local buffer = nn.Sequential()
        for i = 1, upsampler:size() do
            if not torch.type(upsampler:get(i)):lower():find('relu') then
                buffer:add(upsampler:get(i):clone())
            end
        end
        upsampler = buffer:clone()
    end
    model:add(upsampler)
    model:add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    model:reset()

    return model
end

return createModel