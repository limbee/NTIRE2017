require 'nn'
require 'model/common'

local function createModel(opt)
    local scale = opt.scale
    opt.nOut = #scale
    opt.isSwap = true
    
    local addbn = false
    local addrelu = false

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local body = seq()
    for i = 1, opt.nResBlock do
        body:add(resBlock(opt.nFeat, addbn, actParams))
    end

    local model = seq():add(conv(opt.nChannel, opt.nFeat, 3, 3, 1, 1, 1, 1))
    local deblurCat = concat()
    for i = 1, #scale do
        local deblurSeq = seq()
        for j = 1, 4 do
            deblurSeq:add(addSkip(seq()
                :add(conv(opt.nFeat, opt.nFeat, 5, 5, 1, 1, 2, 2))
                :add(act(actParams))
                :add(conv(opt.nFeat, opt.nFeat, 5, 5, 1, 1, 2, 2))))
        end
        deblurCat:add(deblurSeq)
    end
    model:add(deblurCat)

    body:add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
    model:add(addSkip(body))
    
    local cat = concat()
    for i = 1, #scale do
        cat:add(seq()
            :add(upsample_wo_act(scale[i], opt.upsample, opt.nFeat))
            :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1)))
    end

    model:add(cat)

    return model
end

return createModel
