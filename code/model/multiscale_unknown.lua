require 'nn'
require 'model/common'

local function createModel(opt)
    local scale = opt.scale
    opt.nOut = #scale
    opt.isSwap = true
    
    local addbn = false
    local addrelu = false

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    print('\t Load pre-trained Multiscale model and add deblur module')
    assert(opt.preTrained ~= '.', 'Please specify -preTrained option')
    local refModel = torch.load(opt.preTrained)
    local branch = nn.ParallelTable()
    for i = 1, #scale do
        local deblur = seq()
        for j = 1, 2 do
            deblur:add(addSkip(seq()
                :add(conv(opt.nFeat, opt.nFeat, 5, 5, 1, 1, 2, 2))
                :add(act(actParams))
                :add(conv(opt.nFeat, opt.nFeat, 5, 5, 1, 1, 2, 2))))
        end
        branch:add(deblur) 
    end
    refModel:insert(branch, 3)

    return refModel
end

return createModel
