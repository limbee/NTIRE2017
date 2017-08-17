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

    local refModel = nil 
	if opt.preTrained == '.' then
		refModel = require('model/multiscale')(opt)
	else
		refModel = torch.load(opt.preTrained)
	end

    local branch = nn.ParallelTable()
    local scaleRes = (opt.scaleRes and opt.scaleRes ~= 1) and opt.scaleRes or false

	for i = 1, #scale do
        local deblur = seq()
        for j = 1, 2 do
			if scaleRes then
				deblur:add(addSkip(seq()
					:add(conv(opt.nFeat, opt.nFeat, 5, 5, 1, 1, 2, 2))
					:add(act(actParams))
					:add(conv(opt.nFeat, opt.nFeat, 5, 5, 1, 1, 2, 2))
					:add(mulc(scaleRes, false))))
			else
				deblur:add(addSkip(seq()
					:add(conv(opt.nFeat, opt.nFeat, 5, 5, 1, 1, 2, 2))
					:add(act(actParams))
					:add(conv(opt.nFeat, opt.nFeat, 5, 5, 1, 1, 2, 2))))
			end
        end
        branch:add(deblur) 
    end
	if opt.preTrained == '.' then
		refModel:insert(branch, 2)
	else
    	refModel:insert(branch, 3)
	end

    return refModel
end

return createModel
