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

    local function convBlock(nFeat)
        local s = nn.Sequential()
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
            :add(relu(true))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
        return seq()
            :add(concat()
                :add(s)
                :add(id()))
            :add(cadd(true))
    end
   
    local body = seq()
    for i=1,opt.nResBlock do
        body:add(convBlock(opt.nFeat))
    end
    body:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))
    body:add(bnorm(opt.nFeat))

    model = seq()
        :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
        :add(relu(true))
        :add(concat()
            :add(body)
            :add(id()))
        :add(cadd(true))

    if opt.upsample == 'full' then
        if opt.scale == 2 then
            model:add(deconv(opt.nFeat,opt.nFeat, 4,4, 2,2, 1,1))
            model:add(relu(true))
        elseif opt.scale == 3 then
            model:add(deconv(opt.nFeat,opt.nFeat, 6,6, 3,3, 2,2, 1,1))
            model:add(relu(true))
        elseif opt.scale == 4 then
            model:add(deconv(opt.nFeat,opt.nFeat, 8,8, 4,4, 2,2))
            model:add(relu(true))
        end
    elseif opt.upsample == 'shuffle' then -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        if opt.scale == 2 then
            model:add(conv(opt.nFeat,4*opt.nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(relu(true))
        elseif opt.scale == 3 then
            model:add(conv(opt.nFeat,9*opt.nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(3))
            model:add(relu(true))
        elseif opt.scale == 4 then
            model:add(conv(opt.nFeat,4*opt.nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(relu(true))
            model:add(conv(opt.nFeat,4*opt.nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(relu(true))
        end

        -- model:add(pad(2,1,3)) -- 1 pixel padding at bottom
        -- model:add(pad(3,1,3)) -- 1 pixel padding at right
        -- model:add(conv(opt.nFeat,4*upFeat, 4,4, 1,1, 1,1)) -- even size of kernel removes checkerboard artifacts
        -- model:add(shuffle(2))
        -- model:add(relu(true))
    end

    model:add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    return model
end

return createModel
