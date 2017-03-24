require 'nn'
require 'cunn'

require('loss/CharbonnierCriterion')
require('loss/KernelCriterion')
require('loss/GradPriorCriterion')
require('loss/FourierDistCriterion')

local function getLoss(opt)
    local criterion = nn.MultiCriterion()

    if opt.abs > 0 then
        local absLoss = nn.AbsCriterion()
        absLoss.sizeAverage = true
        criterion:add(absLoss, opt.abs)
    end
    if opt.chbn > 0 then
        local chbnLoss = nn.CharbonnierCriterion(true, 0.001)
        criterion:add(chbnLoss, opt.chbn)
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
    if opt.grad2 > 0 then
        local kernel2 = torch.CudaTensor{{{0, 0, 0}, {1, -2, 1}, {0, 0, 0}}, {{0, 1, 0}, {0, -2, 0}, {0, 1, 0}}}
        local grad2Loss = nn.KernelCriterion(opt, kernel2)
        criterion:add(grad2Loss, opt.grad2)
    end
    if opt.gradPrior > 0 then
        local gradPriorLoss = nn.GradPriorCriterion(opt)
        criterion:add(gradPriorLoss, opt.gradPrior)
    end
    if opt.fd > 0 then
        local fdLoss = nn.FilteredDistCriterion(opt, true)
        criterion:add(fdLoss, opt.fd)
    end

    if opt.nOut > 1 then
        local pCri = nn.ParallelCriterion()
        for i = 1, opt.nOut do
            pCri:add(criterion:clone())
        end
        return pCri:cuda()
    else
        return criterion:cuda()
    end
end

return getLoss
