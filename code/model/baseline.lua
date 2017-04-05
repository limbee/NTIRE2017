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
    for i = 1, opt.nResBlock do
        body:add(resBlock(opt.nFeat, false, actParams, opt.scaleRes))
    end
    body:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))

    ret = seq():add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
    if opt.degrade == 'bicubic' then
        ret:add(addSkip(body, true))
    else
        ret:add(body)
    end
    ret:add(upsample_wo_act(opt.scale[1], opt.upsample, opt.nFeat))
        :add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))
    
    return ret
end

return createModel
