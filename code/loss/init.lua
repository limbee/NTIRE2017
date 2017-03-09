require 'nn'
require 'cunn'
require 'tvnorm-nn'

local function getLoss(opt)
    local criterion = nn.MultiCriterion()

    if (opt.abs > 0) then
        local absLoss = nn.ABSCriterion(true)
        criterion:add(absLoss, opt.abs)
    end
    if (opt.chbn > 0) then
        require('loss/CharbonnierCriterion')
        local chbnLoss = nn.CharbonnierCriterion(true, 0.001)
        criterion:add(chbnLoss, opt.chbn)
    end
    if (opt.smoothL1 > 0) then
        local smoothL1 = nn.smoothL1Criterion(true)
        criterion:add(smoothL1, opt.smoothL1)
    end
    if (opt.mse > 0) then
        local mseLoss = nn.MSECriterion(true)
        criterion:add(mseLoss, opt.mse)
    end
    if (opt.ssim > 0) then
        require('loss/SSIMCriterion')
        local ssimLoss = nn.SSIMCriterion()
        criterion:add(ssimLoss, opt.ssim)
    end
    if (opt.fd > 0) then
        require('loss/FourierDistCriterion')
        local fdLoss = nn.FilteredDistCriterion(opt.filter_wc, opt.filter_type)
        criterion:add(hfLoss, opt.fd)
    end
    if (opt.netType == 'bandnet') then
        require('loss/BandCriterion')
        local bandLoss = nn.BandCriterion(opt.netwc, true)
        criterion:add(bandLoss, opt.mse)
    end
        
    return criterion:cuda()
end

return getLoss