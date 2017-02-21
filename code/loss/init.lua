require 'nn'
require 'cunn'
require 'tvnorm-nn'
require 'loss/PerceptualLoss'
require 'loss/AdversarialLoss'

local function getLoss(opt)
    local criterion = nn.MultiCriterion()
    local iAdvLoss = 1

    -- 1. content loss
    if opt.abs > 0 then
        local abs_loss = nn.ABSCriterion()
        abs_loss.sizeAverage = true
        criterion:add(abs_loss, opt.abs)
        iAdvLoss = iAdvLoss + 1
    end
    if opt.smoothL1 > 0 then
        local smoothL1 = nn.smoothL1Criterion()
        smoothL1.sizeAverage = true
        criterion:add(smoothL1, opt.smoothL1)
        iAdvLoss = iAdvLoss + 1
    end
    if opt.mse > 0 then
        local mse_loss = nn.MSECriterion()
        mse_loss.sizeAverage = true
        criterion:add(mse_loss, opt.mse)
        iAdvLoss = iAdvLoss + 1
    end
    if opt.perc > 0 then
        local perceptual_loss = nn.PerceptualLoss(opt)
        criterion:add(perceptual_loss, opt.perc)
        iAdvLoss = iAdvLoss + 1
    end

    -- 2. adversarial loss
    if opt.adv > 0 then
        local adv_loss = nn.AdversarialLoss(opt)
        --local weights = torch.Tensor(opt.batchSize):fill(1/torch.log(2))
        --local adv_loss = nn.BCECriterion(weights) 
        criterion:add(adv_loss, opt.adv)
    end

    -- 3. regularization loss (total variation)
    if opt.tv > 0 then
        local tv_loss = nn.SpatialTVNormCriterion()
        criterion:add(tv_loss, opt.tv)
    end

    if opt.nGPU > 0 then
        criterion = criterion:cuda();
    end

    criterion.iAdvLoss = iAdvLoss    
    return criterion
end

return getLoss