require 'nn'
require 'cunn'
require 'cudnn'

local function getModel(opt)
    local model

    if opt.load then
        local modelPath
        if opt.startEpoch == 0 then
            modelPath = paths.concat(opt.save,'model','model_latest.t7')
        else
            modelPath = paths.concat(opt.save, 'model', 'model_' .. opt.startEpoch - 1 .. '.t7')
        end
        assert(paths.filep(modelPath), 'Saved model not found in ' .. opt.save)
        print('Resuming model from ' .. modelPath)
        model = torch.load(modelPath)
        if torch.type(model) == 'nn.DataParallelTable' then
            model = model:get(1)
        end
    else
        if opt.preTrained ~= 'nil' then
            print('Loading pre-trained model from: ' .. opt.preTrained)
            model = torch.load(opt.preTrained)
        else
            print('Creating model from file: models/' .. opt.netType .. '.lua')
            model = require('model/' .. opt.netType)(opt)
        end

    end

    local meanVec = torch.Tensor({0.4488, 0.4371, 0.4040}):mul(opt.mulImg)
    local stdMat = torch.Tensor({{0.2845, 0, 0},
                                {0, 0.2701, 0},
                                {0, 0, 0.2920}}):mul(opt.mulImg)
    if not opt.load then
        -- Assumes R,G,B order
        if opt.subMean then
            if torch.type(model) ~= 'nn.Sequential' then
                model = nn.Sequential():add(model) -- in case the outermost shell is not a nn.Sequential
            end

            local subMean = nn.SpatialConvolution(3, 3, 1, 1)
            subMean.weight:copy(torch.eye(3, 3):reshape(3, 3, 1, 1))
            subMean.bias:copy(torch.mul(meanVec, -1))
            local addMean = nn.SpatialConvolution(3, 3, 1, 1)
            addMean.weight:copy(torch.eye(3,3):reshape(3, 3, 1, 1))
            addMean.bias:copy(meanVec)
            
            if opt.divStd then
                local divStd = nn.SpatialConvolution(3, 3, 1, 1):noBias()
                divStd.weight:copy(torch.inverse(stdMat):reshape(3, 3, 1, 1))
                local mulStd = nn.SpatialConvolution(3, 3, 1, 1):noBias()
                mulStd.weight:copy(stdMat:reshape(3, 3, 1, 1))

                model:insert(divStd,1)
                model:insert(mulStd)
            end
            model:insert(subMean,1)
            model:insert(addMean)
        else
            assert(not opt.divStd, 'Please set the -subMean option to true')
            opt.trainNormLayer = false
        end
    end

    model = cudnn.convert(model,cudnn)

    local fixBias, fixStd = false, false
    if not opt.load and opt.subMean and not opt.trainNormLayer then
        fixBias = true
    elseif opt.load and model:size() >= 3 and not opt.trainNormLayer then
        local subMean_candidate = model:get(1):clone():float()
        local addMean_candidate = model:get(model:size()):clone():float()
        if torch.type(subMean_candidate):find('SpatialConvolution') and 
            torch.type(addMean_candidate):find('SpatialConvolution') then
            fixBias = torch.equal(subMean_candidate.weight, torch.eye(3, 3):reshape(3, 3, 1, 1)) and
                torch.equal(subMean_candidate.bias, torch.mul(meanVec, -1)) and
                torch.equal(addMean_candidate.weight, torch.eye(3, 3):reshape(3, 3, 1, 1)) and
                torch.equal(addMean_candidate.bias, meanVec)
        end
    end
    if fixBias and model:size() >= 5 then
        local divStd_candidate = model:get(2):clone():float()
        local mulStd_candidate = model:get(model:size()-1):clone():float()
        if torch.type(divStd_candidate):find('SpatialConvolution') and 
            torch.type(mulStd_candidate):find('SpatialConvolution') then
            fixStd = divStd_candidate.bias == nil and
                mulStd_candidate.bias == nil and
                torch.equal(divStd_candidate.weight, torch.inverse(stdMat):reshape(3, 3, 1, 1)) and
                torch.equal(mulStd_candidate.weight, stdMat:reshape(3, 3, 1, 1))
        end
    end
    
    if fixBias then
        model:get(1).accGradParameters = function(input,gradOutput,scale) return end
        model:get(#model).accGradParameters = function(input,gradOutput,scale) return end
        if fixStd then
            model:get(2).accGradParameters = function(input,gradOutput,scale) return end
            model:get(#model-1).accGradParameters = function(input,gradOutput,scale) return end
        end
    end

    model:cuda()
    cudnn.fastest = true
    cudnn.benchmark = true

    if opt.nGPU > 1 then
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

    return model
end

return getModel