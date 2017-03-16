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
        for i = 1, (opt.nConv - 4) / 4 do
            body = addSkip(seq()
                :add(cbrcb(nFeat))
                :add(body)
                :add(cbrcb(nFeat)))
        end
    elseif opt.modelVer == 2 then
        body = addSkip(seq()
            :add(cbrcb(nFeat)) -- It was totally wrong...
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
    -- Version 1 slightly wins among the 2 and 3.
    -- However, none of them succeeded in beating the resnet.
    -------------------------------------------------

    elseif opt.modelVer == 4 then
        body = resBlock(nFeat)
        for i = 1, (opt.nConv - 4) / 2 do
            body = addSkip(seq()
                :add(cbrcb(nFeat))
                :add(body))
        end
    elseif opt.modelVer == 5 then
        bocy = resBlock(nFeat)
        for i = 1, (opt.nConv - 4 ) 2 do
            body = addSkip(seq()
                :add(resBlock(nFeat))
                :add(body))
        end 
    elseif opt.modelVer == 6 then
        body = resBlock(nFeat)
        for i = 1, (opt.nConv - 4) / 4 do
            body = addSkip(seq()
                :add(cbrcb(nFeat))
                :add(relu)
                :add(cbrcb(nFeat))
                :add(body))
        end
    elseif opt.modelVer == 7 then
        body = resBlock(nFeat)
        for i = 1, (opt.nConv - 4) / 4 do
            body = addSkip(seq()
                :add(resBlock(nFeat))
                :add(resBlock(nFeat))
                :add(body))
        end
    else

    end



    model = seq()
        :add(conv(opt.nChannel,nFeat, 3,3, 1,1, 1,1))
        :add(relu(true))
        :add(body) -- 'body' itself has a residual skip connection
        -- :add(concat()
        --     :add(body)
        --     :add(id()))
        :add(cadd(true))

    model:add(require 'model/upsample'(opt))

    model:add(conv(nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    return model
end

return createModel