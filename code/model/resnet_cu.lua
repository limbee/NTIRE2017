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
    
    print('\t Load pre-trained SRResnet and change upsampler')
    local refModel = torch.load(opt.preTrained)
    local nSeq = 0
    local model = nn.Sequential()
    for i = 2, (refModel:size() - 1) do
        local subModelName = refModel:get(i).__typename
        if subModelName:find('Sequential') then
            nSeq = nSeq + 1
        end
        if (nSeq > 1) and subModelName:find('Sequential') then
            model:add(upsample_wo_act(opt.scale, opt.upsample, opt.nFeat))
        else
            model:add(refModel:get(i):clone())
        end
    end

    return model
end

return createModel
