require 'nn'

local function createModel(opt)
    local conv = nn.SpatialConvolution
    local deconv = nn.SpatialFullConvolution
    local shuffle = nn.PixelShuffle
    local relu = nn.ReLU

    local upsample = nn.Sequential()
    if opt.upsample == 'full' then
        if opt.scale == 2 then
            upsample:add(deconv(opt.nFeat,opt.nFeat, 6,6, 2,2, 2,2))
            upsample:add(relu(true))
        elseif opt.scale == 3 then
            upsample:add(deconv(opt.nFeat,opt.nFeat, 9,9, 3,3, 3,3))
            upsample:add(relu(true))
        elseif opt.scale == 4 then
            upsample:add(deconv(opt.nFeat,opt.nFeat, 6,6, 2,2, 2,2))
            upsample:add(relu(true))
            upsample:add(deconv(opt.nFeat,opt.nFeat, 6,6, 2,2, 2,2))
            upsample:add(relu(true))
        end
    elseif opt.upsample == 'shuffle' then -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        if opt.scale == 2 then
            upsample:add(conv(opt.nFeat,4*opt.nFeat, 3,3, 1,1, 1,1))
            upsample:add(shuffle(2))
            upsample:add(relu(true))
        elseif opt.scale == 3 then
            upsample:add(conv(opt.nFeat,9*opt.nFeat, 3,3, 1,1, 1,1))
            upsample:add(shuffle(3))
            upsample:add(relu(true))
        elseif opt.scale == 4 then
            upsample:add(conv(opt.nFeat,4*opt.nFeat, 3,3, 1,1, 1,1))
            upsample:add(shuffle(2))
            upsample:add(relu(true))
            upsample:add(conv(opt.nFeat,4*opt.nFeat, 3,3, 1,1, 1,1))
            upsample:add(shuffle(2))
            upsample:add(relu(true))
        end
    end
    return upsample
end

return createModel