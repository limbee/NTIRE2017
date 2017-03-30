require 'nn'
require 'model/common'

local function createModel(opt)

    if opt.modelVer == 1 then
        print('\t ResNet version 1!')
        print('\t Original ResNet')
    elseif opt.modelVer == 2 then
        print('\t ResNet version 2!')
        print('\t ResNet w/o Batch Normalization')
    elseif opt.modelVer == 3 then
        print('\t ResNet version 3!')
        print('\t ResNet w/o Batch Normalization')
        print('\t ResNet w/o ReLU at front & upsampling')
    else
        error("Unknown ResNet version!")
    end

    local addbn = opt.modelVer == 1

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local body = seq()
    for i=1,opt.nResBlock do
        body:add(resBlock(opt.nFeat, addbn, actParams))
    end
    body:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))

    local model
    if opt.modelVer == 1 then
        body:add(bnorm(opt.nFeat))
        model = seq()
            :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
            :add(act(actParams, opt.nFeat))
            :add(addSkip(body))
            :add(upsample(opt.scale[1], opt.upsample, opt.nFeat, actParams))
            :add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    elseif opt.modelVer == 2 then
        model = seq()
            :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
            :add(act(actParams, opt.nFeat))
            :add(addSkip(body))
            :add(upsample(opt.scale[1], opt.upsample, opt.nFeat, actParams))
            :add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    elseif opt.modelVer == 3 then
        model = seq()
            :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
            :add(addSkip(body))
            :add(upsample_wo_act(opt.scale[1], opt.upsample, opt.nFeat))
            :add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))
    end

    return model
end

return createModel