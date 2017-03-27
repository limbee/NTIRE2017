require 'nn'
require 'model/common'

local function createModel(opt)
    local scale = opt.scale
    opt.nOut = #scale

    local addbn = false
    local addrelu = false

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local front = math.floor(opt.nResBlock * opt.mobranch)
    local back = opt.nResBlock - front

    local body = seq()
    for i = 1, front do
        body:add(resBlock(opt.nFeat, addbn, actParams))
    end

    local model = seq():add(conv(opt.nChannel, opt.nFeat, 3, 3, 1, 1, 1, 1))
    if front > 0 then
        model:add(addSkip(body))
    end
    if back == 0 then
        model:add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1))
    end

    local cat = concat()
    for i = 1, #scale do
        if back > 0 then
            local backRes = seq()
            for j = 1, back do
                backRes:add(resBlock(opt.nFeat, addbn, actParams))
            end
            cat:add(backRes
                :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1))
                :add(upsample_wo_act(scale[i], opt.upsample, opt.nFeat))
                :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1)))
        else
            cat:add(seq()
                :add(upsample_wo_act(scale[i], opt.upsample, opt.nFeat))
                :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1)))
        end
    end    
    model:add(cat)

    return model
end

return createModel
