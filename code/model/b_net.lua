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
    local sc = opt.scale
    local nFeat = opt.nFeat

    if opt.modelVer == 1 then    

        local body = seq()
        for i = 1, opt.nResBlock do
            body:add(resBlock(opt.nFeat, false, actParams))
        end
        body:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))

        model = seq()
            :add(concat()
                :add(deconv(nCh, nCh, 3 * sc, 3 * sc, sc, sc, sc, sc))
                :add(seq()
                    :add(conv(nCh, nFeat, 3,3, 1,1, 1,1))
                    :add(body)
                    :add(upsample_wo_act(sc, opt.upsample, nFeat))
                    :add(conv(nFeat, nCh, 3,3, 1,1, 1,1))))
            :add(cadd(true))

    elseif opt.modelVer == 2 then
        local head = conv(opt.nChannel, opt.nFeat, 3,3, 1,1, 1,1)
      
        model = seq()
    end


    return model
end

return createModel