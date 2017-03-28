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

    if back == 0 then
        body:add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
        model:add(addSkip(body))

        local cat = concat()
        for i = 1, #scale do
            cat:add(seq()
                :add(upsample_wo_act(scale[i], opt.upsample, opt.nFeat))
                :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1)))
        end

        model:add(cat)
    else
        local cat = concat()
        for i = 1, #scale do
            local bSeq = seq()
            for j = 1, back do
                bSeq:add(resBlock(opt.nFeat, addbn, actParams))
            end
            bSeq:add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
            cat:add(bSeq)
        end

        model:add(seq():add(concat()
                :add(id())
                :add(body
                    :add(cat))))
            :add(nn.MultiSkipAdd(true))

        local par = nn.ParallelTable()
        for i = 1, #scale do
            par:add(seq()
                :add(upsample_wo_act(scale[i], opt.upsample, opt.nFeat))
                :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1)))
        end
        model:add(par)
    end

    return model
end

return createModel
