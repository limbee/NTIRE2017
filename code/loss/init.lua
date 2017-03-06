require 'nn'
require 'cunn'
require 'tvnorm-nn'

local function getLoss(opt)
    local criterion = nn.MultiCriterion()

    -- 1. content loss
    if (opt.abs > 0) then
        local abs_loss = nn.ABSCriterion()
        abs_loss.sizeAverage = true
        criterion:add(abs_loss, opt.abs)
    end
    if (opt.chbn > 0) then
        dofile('CharbonnierCriterion.lua')
        local chbn_loss = nn.CharbonnierCriterion(true, 0.001)
        criterion:add(chbn_loss, opt.chbn)
    end
    if (opt.smoothL1 > 0) then
        local smoothL1 = nn.smoothL1Criterion()
        smoothL1.sizeAverage = true
        criterion:add(smoothL1, opt.smoothL1)
    end
    if (opt.mse > 0) then
        local mse_loss = nn.MSECriterion()
        mse_loss.sizeAverage = true
        criterion:add(mse_loss, opt.mse)
    end
    if (opt.ssim > 0) then
        dofile('SSIMCriterion.lua')
        local ssim_loss = nn.SSIMCriterion()
        criterion:add(ssim_loss, opt.ssim)
    end
    if (opt.fd > 0) then
        dofile('FourierDistCriterion.lua')
        local fd_loss = nn.FilteredDistCriterion(opt.filter_wc, opt.filter_type)
        criterion:add(hf_loss, opt.fd)
    end
    return criterion:cuda()
end

return getLoss