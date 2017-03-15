require 'nn'
require 'cunn'

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

    local function resBlock(nFeat)
        return seq()
            :add(concat()
                :add(cbrcb(nFeat))
                :add(id()))
            :add(cadd(true))
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
        for i = 1, (opt.nConv - 2) / 4 do
            body = addSkip(seq()
                :add(cbrcb(nFeat))
                :add(body)
                :add(cbrcb(nFeat)))
        end
    elseif opt.modelVer == 2 then
        body = addSkip(seq()
            :add(cbrcb(nFeat))
            :add(cbrcb(nFeat)))
        for i = 1, (opt.nConv - 2) / 8 do
            body = addSkip(seq()
                :add(cbrcb(nFeat))
                :add(cbrcb(nFeat))
                :add(body)
                :add(cbrcb(nFeat))
                :add(cbrcb(nFeat)))
        end
    elseif opt.modelVer == 3 then
        body = resBlock(nFeat)
        for i = 1, (opt.nConv - 2) / 4 do
            body = addSkip(seq()
                :add(resBlock(nFeat))
                :add(body)
                :add(resBlock(nFeat)))
        end
    else

    end



    model = seq()
        :add(conv(opt.nChannel,nFeat, 3,3, 1,1, 1,1))
        :add(relu(true))
        :add(concat()
            :add(body)
            :add(id()))
        :add(cadd(true))

    if opt.upsample == 'full' then
        if opt.scale == 2 then
            model:add(deconv(nFeat,nFeat, 4,4, 2,2, 1,1))
            model:add(relu(true))
        elseif opt.scale == 3 then
            model:add(deconv(nFeat,nFeat, 6,6, 3,3, 2,2, 1,1))
            model:add(relu(true))
        elseif opt.scale == 4 then
            model:add(deconv(nFeat,nFeat, 8,8, 4,4, 2,2))
            model:add(relu(true))
        end
    elseif opt.upsample == 'shuffle' then -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        if opt.scale == 2 then
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(relu(true))
        elseif opt.scale == 3 then
            model:add(conv(nFeat,9*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(3))
            model:add(relu(true))
        elseif opt.scale == 4 then
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(relu(true))
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(relu(true))
        end
    end

    model:add(conv(nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    return model
end

return createModel