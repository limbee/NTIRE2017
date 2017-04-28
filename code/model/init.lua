require 'nn'
require 'cunn'
require 'cudnn'

local function getModel(opt)
    local model

    if opt.load then
        local modelPath = paths.concat(opt.save, 'model', 'model_' .. opt.startEpoch - 1 .. '.t7')
        assert(paths.filep(modelPath), 'Saved model not found in ' .. opt.save)
        print('Resuming model from ' .. modelPath)
        require('model/' .. opt.netType)(opt)
        model = torch.load(modelPath)

        if torch.type(model) == 'nn.DataParallelTable' then
            model = model:get(1)
        end
    elseif opt.preTrained ~= '.' then
        print('Loading pre-trained model from: ' .. opt.preTrained)
        if (opt.netType == 'resnet_cu') or (opt.netType == 'multiscale_unknown') then
            model = require('model/' .. opt.netType)(opt)
        else
            model = torch.load(opt.preTrained)
        end
    else
        print('Creating model from file: models/' .. opt.netType .. '.lua')
        model = require('model/' .. opt.netType)(opt)

        if opt.subMean then
            if torch.type(model) ~= 'nn.Sequential' then
                model = nn.Sequential():add(model) -- in case the outermost shell is not a nn.Sequential
            end

            -- Assumes R,G,B order
            local meanVec = torch.Tensor({0.4488, 0.4371, 0.4040}):mul(opt.mulImg)
            local stdMat = torch.Tensor({{0.2845, 0, 0},
                                        {0, 0.2701, 0},
                                        {0, 0, 0.2920}}):mul(opt.mulImg)

            local subMean = nn.SpatialConvolution(3, 3, 1, 1)
            subMean.weight:copy(torch.eye(3, 3):reshape(3, 3, 1, 1))
            subMean.bias:copy(torch.mul(meanVec, -1))
            local addMean = nn.SpatialConvolution(3, 3, 1, 1)
            addMean.weight:copy(torch.eye(3,3):reshape(3, 3, 1, 1))
            addMean.bias:copy(meanVec)
            if not opt.trainNormLayer then
                addMean.accGradParameters = function(input, gradOutput, scale) return end
                subMean.accGradParameters = function(input, gradOutput, scale) return end
            end

            model:insert(subMean, 1)
            if opt.netType:find('multiscale') then
                local pt = nn.ParallelTable()
                for i = 1, #opt.scale do
                    pt:add(addMean:clone())
                end
                model:insert(pt)
            else
                model:insert(addMean)
            end
        else
            assert(not opt.divStd, 'Please set the -subMean option to true')
        end

        model = cudnn.convert(model,cudnn)
    end

	if opt.subMean and opt.dataset == 'imagenet50k' then
		local r, g, b = 0.4785 * opt.mulImg, 0.4571 * opt.mulImg, 0.4072 * opt.mulImg
		local subMeanLayer = model:get(1)
		subMeanLayer.bias:copy(torch.Tensor({-r, -g, -b}))
		if opt.netType:find('multiscale') then
			local addMeanParallel = model:get(model:size())
			for i = 1, #opt.scale do
				addMeanParallel:get(i).bias:copy(torch.Tensor({r, g, b}))
			end
		else
			local addMeanLayer = model:get(model:size())
			addMeanLayer.bias:copy(torch.Tensor({r, g, b}))
		end
	end

	model:cuda()
    cudnn.fastest = true
    cudnn.benchmark = true

    if (opt.nGPU > 1) and (opt.isSwap == false) then
        local gpus = torch.range(1, opt.nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
            :add(model, gpus)
            :threads(function()
                local cudnn = require 'cudnn'
                cudnn.fastest, cudnn.benchmark = fastest, benchmark
            end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end

    if opt.printModel then
        print(model)
    end
   	 
    return model
end

return getModel
