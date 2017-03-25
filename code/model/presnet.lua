require 'nn'
require 'model/common'

local function createModel(opt)
    local addbn = opt.modelVer == 1
    local addrelu = (opt.modelVer == 1) or (opt.modelVer == 2)

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local model = seq()
        :add(conv(opt.nChannel, opt.nFeat, 3, 3, 1, 1, 1, 1))
    
    local body = seq()
    for i = 1, (2 * opt.nResBlock / 3) do
        body:add(resBlock(opt.nFeat, addbn, actParams))
    end
    
    local branch = concat()

    local shallow = seq()
    local deep = seq()
    for i = 1, opt.nResBlock / 3 do
        deep:add(resBlock(opt.nFeat, addbn, actParams))
    end

    shallow:add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
    deep:add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
    body:add(concat()
        :add(shallow)
        :add(deep))
    model:add(concat()
            :add(id())
            :add(body))
        :add(nn.FlattenTable())

    model:add(concat()
        :add(seq()
            :add(nn.NarrowTable(1, 2))
            :add(cadd(true))
            :add(upsample_wo_act(opt.scale, opt.upsample, opt.nFeat))
            :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1)))
        :add(seq()
            :add(concat()
                :add(nn.SelectTable(1))
                :add(nn.SelectTable(3)))
            :add(cadd(true))
            :add(upsample_wo_act(opt.scale, opt.upsample, opt.nFeat))
            :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1))))
    
    return model
end

return createModel
