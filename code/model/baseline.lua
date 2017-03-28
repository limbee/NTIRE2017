require 'nn'
require 'model/common'

local function createModel(opt)

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local body = seq()
    for i=1,opt.nResBlock do
        body:add(resBlock(opt.nFeat, false, actParams))
    end
    body:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))

    return seq()
        :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
        :add(addSkip(body, true))
        :add(upsample_wo_act(opt.scale, opt.upsample, opt.nFeat))
        :add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

end

return createModel