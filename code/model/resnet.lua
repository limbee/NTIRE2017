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

    model:add(require 'model/upsample'(opt))



    model:add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    return model
end

return createModel