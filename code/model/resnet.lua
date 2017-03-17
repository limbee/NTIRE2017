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
    local addrelu = (opt.modelVer == 1) or (opt.modelVer == 2)

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
    body:add(bnorm(opt.nFeat))



    local model = seq()
    if opt.modelVer == 1 then
        body:add(bnorm(opt.nFeat))
        model = seq()
            :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
            :add(act(actParams, opt.nFeat))
            :add(addSkip(body))
            :add(upsample(opt.scale, opt.upsample, opt.nFeat, actParams))
            :add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    elseif opt.modelVer == 2 then
        model = seq()
            :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
            :add(act(actParams, opt.nFeat))
            :add(addSkip(body))
            :add(upsample(opt.scale, opt.upsample, opt.nFeat, actParams))
            :add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    elseif opt.modelVer == 3 then
        local upsampler = upsample(opt.scale, opt.upsample, opt.nFeat, actParams)
        local buffer = nn.Sequential()
        for i = 1, upsampler:size() do
            local name = torch.type(upsampler:get(i)):lower()
            if not (name:find('relu') or name:find('nn.elu')) then
                buffer:add(upsampler:get(i):clone())
            end
        end
        upsampler = buffer:clone()
        buffer = nil
        collectgarbage()
        collectgarbage()

        model = seq()
            :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
            :add(addSkip(body))
            :add(upsampler)
            :add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))
    end

    return model
end

return createModel