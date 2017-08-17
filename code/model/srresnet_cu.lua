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
    assert(opt.preTrained ~= '.', 'Please specify -preTrained option')
    local refModel = torch.load(opt.preTrained)
    local nSeq = 0
    local model = nn.Sequential()
    for i = 1, refModel:size() do
        local subModel = refModel:get(i)
        local subModelName = subModel.__typename
        local isShuffler = false
        if subModelName:find('Sequential') then
            for j = 1, subModel:size() do
                if subModel:get(j).__typename:find('PixelShuffle') then
                    isShuffler = true
                    break
                end
            end
        end
        if isShuffler then
            print('\t Changing upsample layers')
            --model:add(upsample_wo_act(opt.scale[1], opt.upsample, opt.nFeat))
        	model:add(upsample(opt.scale[1], opt.upsample, opt.nFeat, actParams))
		else
            model:add(refModel:get(i):clone())
        end
    end

    return model
end

return createModel
