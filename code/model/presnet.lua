require 'nn'
require 'model/common'

local function createModel(opt)
    local scale = opt.scale[1]
    opt.nOut = 2
    
    local addbn = false
    local addrelu = false

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local model = seq()
        :add(conv(opt.nChannel, opt.nFeat, 3, 3, 1, 1, 1, 1))
    
    local nFront = 2 * opt.nResBlock / 3
    local nBack = opt.nResBlock - nFront

    local body = seq()
    for i = 1, nFront do
        body:add(resBlock(opt.nFeat, addbn, actParams))
    end

    local deep = seq()
    for i = 1, nBack do
        deep:add(resBlock(opt.nFeat, addbn, actParams))
    end

    body:add(concat()
        :add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
        :add(deep
            :add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))))
    model:add(concat()
            :add(id())
            :add(body))
        :add(nn.MultiSkipAdd(true))
        :add(nn.ParallelTable()
            :add(seq()
                :add(upsample_wo_act(scale, opt.upsample, opt.nFeat))
                :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1)))
            :add(seq()
                :add(upsample_wo_act(scale, opt.upsample, opt.nFeat))
                :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1))))

    return model
end

return createModel
