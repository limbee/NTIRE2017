require 'nn'
require 'cunn'

local function createModel(opt)
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization
    local shuffle = nn.PixelShuffle
    local pad = nn.Padding

    local function basicblock(nFeat, stride, preActivation)
        local s = nn.Sequential()
        if not preActivation then
            s:add(conv(nFeat,nFeat,3,3,stride,stride,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,1,1,1,1))
            s:add(bnorm(nFeat))
        else
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,stride,stride,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,1,1,1,1))
        end

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(nn.Identity()))
            :add(nn.CAddTable(true))
    end

    local function bottleneck(nFeat, stride, preActivation)
        local s = nn.Sequential()
        if not preActivation then
            s:add(conv(nFeat,nFeat,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,stride,stride,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,1,1))
            s:add(bnorm(nFeat))
        else
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,stride,stride,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,1,1))
        end

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(nn.Identity()))
            :add(nn.CAddTable(true))
    end
    
    local preActivation = opt.pre_act
    local conv_block = opt.bottleneck and bottleneck or basicblock

    local head = nn.Sequential()
        :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
        :add(relu(true))

    local model = nn.Sequential()
    for i=1,opt.nResBlock do
        model:add(conv_block(opt.nFeat, 1, preActivation))
    end
    if opt.netType == 'preResNet' then
        head:remove(2) -- remove relu (duplicated)
        model:add(bnorm(opt.nFeat))
    end
    model:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))
    model:add(bnorm(opt.nFeat))

    model = nn.Sequential()
        :add(head)
        :add(nn.ConcatTable()
            :add(model)
            :add(nn.Identity()))
        :add(nn.CAddTable(true))

    if opt.upsample == 'full' then
        model:add(nn.SpatialFullConvolution(opt.nFeat,opt.nFeat, 4,4, 2,2, 1,1))
        model:add(relu(true))
        --model:add(nn.SpatialFullConvolution(opt.nFeat,opt.nFeat, 4,4, 2,2, 1,1))
        --model:add(relu(true))
    elseif opt.upsample == 'shuffle' then -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        local upFeat = opt.nFeat * 2 * 2
        model:add(pad(2,1,3))
        model:add(pad(3,1,3))
        model:add(conv(opt.nFeat,upFeat, 4,4, 1,1, 1,1))
        model:add(shuffle(2))
        model:add(relu(true))
        --model:add(pad(2,1,3))
        --model:add(pad(3,1,3))
        --model:add(conv(opt.nFeat,upFeat, 4,4, 1,1, 1,1))
        --model:add(shuffle(2))
        --model:add(relu(true))

    --
    -- Currently bilinear option is not supported
    --
    -- elseif opt.upsample == 'bilinear' then
    --     model:add(nn.SpatialUpSamplingBilinear({owidth=opt.patchSize/2,oheight=opt.patchSize/2}))
    --     model:add(conv(opt.nFeat,opt.nFeat,filt_deconv,filt_deconv,1,1,pad_deconv,pad_deconv))
    --     model:add(relu(true))
    --     model:add(nn.SpatialUpSamplingBilinear({owidth=opt.patchSize,oheight=opt.patchSize}))
    --     model:add(conv(opt.nFeat,opt.nFeat,filt_deconv,filt_deconv,1,1,pad_deconv,pad_deconv))
    --     model:add(relu(true))
    end

    model:add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))

    return model
end

return createModel
