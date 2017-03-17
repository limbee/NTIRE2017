require 'nn'
require 'model/common'

----------------------------------------------------------------------------
-- This file doesn't make the original u-net (Ronneberger 2015)
-- But the overall structure using the skip connection imitated u-net.
-- We don't use any pooling operation here.
----------------------------------------------------------------------------

local function createModel(opt)

    local function resConvBN(nFeat)
        return seq()
            :add(resBlock(nFeat))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
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
        for i = 1, (opt.nConv - 5) / 4 do
            body = addSkip(seq()
                :add(resBlock(nFeat))
                :add(resBlock(nFeat))
                :add(body))
        end
    elseif opt.modelVer == 8 then
        body = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 5) / 8 do
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
