require 'nn'
require 'model/common'

local function createModel(opt)

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval
	
	local scale = opt.scale[1]
    local body = seq()
		:add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
	if scale == 2 then
		body:add(resBlock(opt.nFeat, false, actParams, opt.scaleRes, opt.ipMulc))
		body:add(conv(opt.nFeat,opt.nFeat, 3,3, 2,2, 1,1))
	elseif scale == 3 then
		body:add(resBlock(opt.nFeat, false, actParams, opt.scaleRes, opt.ipMulc))
		body:add(conv(opt.nFeat,opt.nFeat, 3,3, 3,3, 1,1))
	elseif scale == 4 then
		body:add(resBlock(opt.nFeat, false, actParams, opt.scaleRes, opt.ipMulc))
		body:add(conv(opt.nFeat,opt.nFeat, 3,3, 2,2, 1,1))
		body:add(resBlock(opt.nFeat, false, actParams, opt.scaleRes, opt.ipMulc))
		body:add(conv(opt.nFeat,opt.nFeat, 3,3, 2,2, 1,1))
	end
	body:add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    return body
end

return createModel
