require 'nn'
require 'cunn'

require('loss/KernelCriterion')

local function getLoss(opt)
    local criterion = nn.MultiCriterion()

    if opt.abs > 0 then
        local absLoss = nn.AbsCriterion()
        absLoss.sizeAverage = true
        criterion:add(absLoss, opt.abs)
    end
    if opt.smoothL1 > 0 then
        local smoothL1 = nn.SmoothL1Criterion()
        smoothL1.sizeAverage = true
        criterion:add(smoothL1, opt.smoothL1)
    end
    if opt.mse > 0 then
        local mseLoss = nn.MSECriterion()
        mseLoss.sizeAverage = true  
        if opt.mse > 1 then
            criterion = nn.ParallelCriterion()
            criterion.repeatTarget = true
            for j=2,opt.mse do
                criterion:add(mseLoss, opt.mse)
            end
        end
        criterion:add(mseLoss, opt.mse)
    end
    if opt.grad > 0 then
        local kernel = torch.CudaTensor({{{-1, 1}, {0, 0}}, {{-1, 0}, {1, 0}}})
        local gradLoss = nn.KernelCriterion(opt, kernel)
        criterion:add(gradLoss, opt.grad)
    end
    
    if (opt.nOut > 1) and (opt.isSwap == false) then 
        local pCri = nn.ParallelCriterion()
        for i = 1, opt.nOut do
            pCri:add(criterion:clone())
        end
        pCri.repeatTarget = true
        return pCri:cuda()
    else
        return criterion:cuda()
    end
end

return getLoss
