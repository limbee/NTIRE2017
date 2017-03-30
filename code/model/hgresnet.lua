require 'nn'
require 'model/common'

local function createModel(opt)

    print('\t Hourglass Resnet!')
    print('\t Hourglass structure with residual blocks')

    local addbn = false
    local addrelu = false

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local function hourglass(depth)
        if depth == 1 then
            return resBlock(opt.nFeat, addbn, actParams)
        else
            --type 1
            if opt.HGVer == 1 then
                return seq()
                    :add(concat()
                        :add(seq()
                            :add(resBlock(opt.nFeat, addbn, actParams))
                            :add(avgpool(2, 2, 2, 2))
                            :add(hourglass(depth - 1))
                            :add(upsample_wo_act(2, opt.upsample, opt.nFeat))
                            :add(resBlock(opt.nFeat, addbn, actParams)))
                        :add(resBlock(opt.nFeat, addbn, actParams)))
                    :add(cadd(true))
            --type 2
            else
                return seq()
                    :add(resBlock(opt.nFeat, addbn, actParams))
                    :add(concat()
                        :add(seq()
                            :add(avgpool(2, 2, 2, 2))
                            :add(hourglass(depth - 1))
                            :add(upsample_wo_act(2, opt.upsample, opt.nFeat)))
                        :add(resBlock(opt.nFeat, addbn, actParams)))
                    :add(cadd(true))
                    :add(resBlock(opt.nFeat, addbn, actParams))
        end
    end
    
    local body = nn.Sequential()
    for i = 1, opt.nHGBlock do
        body:add(hourglass(opt.HGDepth))
    end
    body:add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))

    local model = nn.Sequential()
        :add(conv(opt.nChannel, opt.nFeat, 3, 3, 1, 1, 1, 1))
        :addSkip(body())
        :add(upsample_wo_act(opt.scale[1], opt.upsample, opt.nFeat))
        :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1))

    return model
end

return createModel
