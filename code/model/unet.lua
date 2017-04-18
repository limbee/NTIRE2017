require 'nn'
require 'model/common'

----------------------------------------------------------------------------
-- This file doesn't make the original u-net (Ronneberger 2015)
-- But the overall structure using the skip connection imitated u-net.
-- We don't use any pooling operation here.
----------------------------------------------------------------------------

local function createModel(opt)

    local nFeat = opt.nFeat

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local function resConvBN(nFeat)
        return seq()
            :add(resBlock(nFeat, true, actParams))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
    end

    local body

    if opt.modelVer == 1 then
        body = resBlock(nFeat, true, actParams)
        for i = 1, (opt.nConv - 4) / 4 do
            body = addSkip(seq()
                :add(cbrcb(nFeat, actParams))
                :add(body)
                :add(cbrcb(nFeat, actParams)))
        end
    elseif opt.modelVer == 2 then
        body = addSkip(seq()
            :add(cbrcb(nFeat, actParams))
            :add(cbrcb(nFeat, actParams)))
        for i = 1, (opt.nConv - 6) / 8 do
            body = addSkip(seq()
                :add(cbrcb(nFeat, actParams))
                :add(cbrcb(nFeat, actParams))
                :add(body)
                :add(cbrcb(nFeat, actParams))
                :add(cbrcb(nFeat, actParams)))
        end
    elseif opt.modelVer == 3 then
        body = resBlock(nFeat, true, actParams)
        for i = 1, (opt.nConv - 4) / 4 do
            body = addSkip(seq()
                :add(resBlock(nFeat, true, actParams))
                :add(body)
                :add(resBlock(nFeat, true, actParams)))
        end
    
    -------------------------------------------------
    -- Version 1 slightly wins the 2 and 3.
    -- However, none of 1 ~ 3 succeeded in beating the resnet.
    -------------------------------------------------

    elseif opt.modelVer == 4 then
        body = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 5) / 2 do
            body = addSkip(seq()
                :add(cbrcb(nFeat, actParams))
                :add(body))
        end
    elseif opt.modelVer == 5 then
        bocy = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 5) / 2 do
            body = addSkip(seq()
                :add(resBlock(nFeat, true, actParams))
                :add(body))
        end 
    elseif opt.modelVer == 6 then
        body = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 5) / 4 do
            body = addSkip(seq()
                :add(cbrcb(nFeat, actParams))
                :add(act(actParams, nFeat))
                :add(cbrcb(nFeat, actParams))
                :add(act(actParams, nFeat))
                :add(body))
        end 
    elseif opt.modelVer == 7 then
        body = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 5) / 4 do
            body = addSkip(seq()
                :add(resBlock(nFeat, true, actParams))
                :add(resBlock(nFeat, true, actParams))
                :add(body))
        end
    elseif opt.modelVer == 8 then
        body = addSkip(resConvBN(nFeat))
        for i = 1, (opt.nConv - 5) / 8 do
            body = addSkip(seq()
                :add(resBlock(nFeat, true, actParams))
                :add(resBlock(nFeat, true, actParams))
                :add(resBlock(nFeat, true, actParams))
                :add(resBlock(nFeat, true, actParams))
                :add(body))
        end


    -------------------------------------------------
    -- modelVer 8 doesn't seem to work better then baseline
    -------------------------------------------------



    elseif opt.modelVer == 9 then
        body = addSkip(seq()
            :add(resBlock(nFeat, false, actParams))
            :add(resBlock(nFeat, false, actParams)))
        for i = 1, (opt.nConv - 12) / 4 do
            body = addSkip(seq()
                :add(crc(nFeat, actParams))
                :add(body)
                :add(crc(nFeat, actParams)))
        end
        body = addSkip(seq()
            :add(crc(nFeat, actParams))
            :add(body)
            :add(crc(nFeat, actParams))
            :add(act(actParams))
            :add(conv(nFeat, nFeat, 3,3, 1,1, 1,1)))

    elseif opt.modelVer == 10 then
        body = addSkip(seq()
            :add(resBlock(nFeat, false, actParams))
            :add(resBlock(nFeat, false, actParams)))
        for i = 1, (opt.nConv - 12) / 4 do
            body = addSkip(seq()
                :add(resBlock(nFeat, false, actParams))
                :add(body)
                :add(resBlock(nFeat, false, actParams)))
        end
        body = addSkip(seq()
            :add(resBlock(nFeat, false, actParams)))
            :add(body)
            :add(resBlock(nFeat, false, actParams)))
            :add(conv(nFeat, nFeat, 3,3, 1,1, 1,1)))

    elseif opt.modelVer == 11 then
        body = addSkip(seq()
            :add(crc(nFeat, actParams))
            :add(act(actParams))
            :add(crc(nFeat, actParams)))
        for i = 1, (opt.nConv - 12) / 4 do
            body = addSkip(seq()
                :add(crc(nFeat, actParams))
                :add(body)
                :add(crc(nFeat, actParams)))
        end
        body = addSkip(seq()
            :add(crc(nFeat, actParams))
            :add(body)
            :add(crc(nFeat, actParams))
            :add(act(actParams))
            :add(conv(nFeat, nFeat, 3,3, 1,1, 1,1)))
    end



    model = seq()
        :add(conv(opt.nChannel,nFeat, 3,3, 1,1, 1,1))
        :add(body)
        :add(upsample_wo_act(opt.scale[1], opt.upsample, nFeat))
        :add(conv(nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    return model
end

return createModel
