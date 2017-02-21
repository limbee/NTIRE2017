require 'nn'
require 'cunn'
require 'cudnn'

local function getModel(opt)
    local model

    if opt.load then
        local modelPath = paths.concat(opt.save,'model','model_latest.t7')
        assert(paths.filep(modelPath), 'Saved model not found in ' .. opt.save)
        print('Resuming model from ' .. modelPath)
        model = torch.load(modelPath)
    else 
        print('Creating model from file: models/' .. opt.netType .. '.lua')
        model = require('model/' .. opt.netType)(opt)
    end

    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    if opt.optnet then
        local optnet = require 'optnet'
        local imsize = opt.dataset == 'imagenet' and 224 or 32
        local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
        optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
    end

    if opt.nGPU > 0 then
        if opt.backend == 'cudnn' then
            cudnn.convert(model,cudnn)
        end
        model:cuda()
        if opt.cudnn == 'fastest' then
            cudnn.fastest = true
            cudnn.benchmark = true
        elseif opt.cudnn == 'deterministic' then
            model:apply(function(m)
                if m.setMode then m:setMode(1, 1, 1) end
            end)
        end
    end

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