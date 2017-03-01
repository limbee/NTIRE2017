require 'nn'
require 'cunn'
require 'tvnorm-nn'

local function getLoss(opt)
    local criterion = nn.MultiCriterion()

    -- 1. content loss
    if opt.abs > 0 then
        local abs_loss = nn.ABSCriterion()
        abs_loss.sizeAverage = true
        criterion:add(abs_loss, opt.abs)
    end
    if opt.smoothL1 > 0 then
        local smoothL1 = nn.smoothL1Criterion()
        smoothL1.sizeAverage = true
        criterion:add(smoothL1, opt.smoothL1)
    end
    if opt.mse > 0 then
        local mse_loss = nn.MSECriterion()
        mse_loss.sizeAverage = true
        criterion:add(mse_loss, opt.mse)
    end

    return criterion:cuda()
end

return getLoss