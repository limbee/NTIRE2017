require 'nn'
require 'cunn'

local function createModel(opt)
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization

    local avg = 0
    local net = nn.Sequential()
    net:add(conv(opt.nChannel, opt.nFeat, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    for i = 1, opt.nLayer - 1 do
        net:add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
        net:add(bnorm(opt.nFeat))
        net:add(relu(true))
    end
    net:add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1))

    local cat = nn.ConcatTable()
    cat:add(net)
    cat:add(nn.Identity())

    local model = nn.Sequential()
    model:add(cat)
    model:add(nn.CAddTable())

    return model
end

return createModel
