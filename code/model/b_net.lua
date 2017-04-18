require 'nn'
require 'model/common'

local function createModel(opt)

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local model

    local nCh = opt.nChannel
    local sc = opt.scale[1]
    local nFeat = opt.nFeat

    -------------------------------------------------------------------------------------
    -- Ver 1 and 2 learns spatial upsampling layer using CNN with kernel size 5*scale
    -- instead of using bicubic interpolation
    -------------------------------------------------------------------------------------

    if opt.modelVer == 1 then

        local body = seq()
        for i = 1, opt.nResBlock do
            body:add(resBlock(opt.nFeat, false, actParams))
        end
        body:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))

        model = seq()
            :add(concat()
                :add(deconv(nCh, nCh, 5 * sc, 5 * sc, sc, sc, 2 * sc, 2 * sc))
                :add(seq()
                    :add(conv(nCh, nFeat, 3,3, 1,1, 1,1))
                    :add(body)
                    :add(upsample_wo_act(sc, opt.upsample, nFeat))
                    :add(conv(nFeat, nCh, 3,3, 1,1, 1,1))))
            :add(cadd(true))
    
    elseif opt.modelVer == 2 then

        local body = seq()
        for i = 1, opt.nResBlock do
            body:add(resBlock(opt.nFeat, false, actParams))
        end
        body:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))

        model = seq()
            :add(concat()
                :add(deconv(nCh, nCh, 5 * sc, 5 * sc, sc, sc, 2 * sc, 2 * sc))
                :add(seq()
                    :add(conv(nCh, nFeat, 3,3, 1,1, 1,1))
                    :add(addSkip(body))
                    :add(upsample_wo_act(sc, opt.upsample, nFeat))
                    :add(conv(nFeat, nCh, 3,3, 1,1, 1,1))))
            :add(cadd(true))

    elseif opt.modelVer == 3 then
    end



    return model
end

return createModel
