require 'nn'
require 'model/common'

local function createModel(opt)

    local addbn = false

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local body = seq()
    for i=1,opt.nResBlock do
        body:add(nextBlock(opt.nextnFeat, opt.nextC, opt.nextF, actParams))
    end
    body:add(conv(opt.nextnFeat,opt.nextnFeat, 3,3, 1,1, 1,1))

    local model = seq()
            :add(conv(opt.nChannel,opt.nextnFeat, 3,3, 1,1, 1,1))
            :add(addSkip(body))
            :add(upsample_wo_act(opt.scale[1], opt.upsample, opt.nextnFeat))
            :add(conv(opt.nextnFeat,opt.nChannel, 3,3, 1,1, 1,1))

    return model
end

return createModel