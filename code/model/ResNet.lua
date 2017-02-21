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
    
    local filt_deconv = opt.filt_deconv
    local filt_recon = opt.filt_recon
    local pad_deconv = (filt_deconv-1)/2
    local pad_deconv_1 = (filt_deconv + 1) % 2
    local pad_deconv_2 = math.floor((filt_deconv-1)/2)
    local pad_recon = (filt_recon-1)/2
    local preActivation = opt.pre_act
    local conv_block = opt.bottleneck and bottleneck or basicblock

    local head = nn.Sequential()
        :add(conv(opt.nChannel,opt.nFeat, filt_recon,filt_recon, 1,1, pad_recon, pad_recon))
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
        model:add(nn.SpatialFullConvolution(opt.nFeat,opt.nFeat, filt_deconv,filt_deconv, 2,2, pad_deconv,pad_deconv, 1,1))
        model:add(relu(true))
        model:add(nn.SpatialFullConvolution(opt.nFeat,opt.nFeat, filt_deconv,filt_deconv, 2,2, pad_deconv,pad_deconv, 1,1))
        model:add(relu(true))
    elseif opt.upsample == 'shuffle' then -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        local upFeat = opt.nFeat * 2 * 2
        if pad_deconv_1 > 0 then
            model:add(pad(2,pad_deconv_1,3))
            model:add(pad(3,pad_deconv_1,3))
        end
        model:add(conv(opt.nFeat,upFeat,filt_deconv,filt_deconv,1,1,pad_deconv,pad_deconv_2))
        model:add(shuffle(2))
        model:add(relu(true))
        if pad_deconv_1 > 0 then
            model:add(pad(2,pad_deconv_1,3))
            model:add(pad(3,pad_deconv_1,3))
        end
        model:add(conv(opt.nFeat,upFeat,filt_deconv,filt_deconv,1,1,pad_deconv,pad_deconv_2))
        model:add(shuffle(2))
        model:add(relu(true))
    -- Currently bilinear option has a bug. owidth and oheight must be adjusted to match the input size at test time.
    elseif opt.upsample == 'bilinear' then
        model:add(nn.SpatialUpSamplingBilinear({owidth=opt.patchSize/2,oheight=opt.patchSize/2}))
        model:add(conv(opt.nFeat,opt.nFeat,filt_deconv,filt_deconv,1,1,pad_deconv,pad_deconv))
        model:add(relu(true))
        model:add(nn.SpatialUpSamplingBilinear({owidth=opt.patchSize,oheight=opt.patchSize}))
        model:add(conv(opt.nFeat,opt.nFeat,filt_deconv,filt_deconv,1,1,pad_deconv,pad_deconv))
        model:add(relu(true))
    end

    model:add(conv(opt.nFeat,opt.nChannel, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))

    --model:insert(nn.Copy(opt.defaultType,opt.operateType),1)
    if opt.normalize then
        local mean = torch.Tensor({0.485,0.456,0.406})
        local subMean = nn.SpatialConvolution(3,3,1,1)
        subMean.weight = torch.eye(3,3):view(3,3,1,1)
        subMean.bias = torch.Tensor(mean):mul(-1)
        local addMean = nn.SpatialConvolution(3,3,1,1)
        addMean.weight = torch.eye(3,3):view(3,3,1,1)
        addMean.bias = torch.Tensor(mean)
        local std = torch.Tensor({0.229,0.224,0.225})
        local divStd = nn.SpatialConvolution(3,3,1,1):noBias()
        divStd.weight = torch.Tensor({{1/std[1],0,0},{0,1/std[2],0},{0,0,1/std[3]}})
        local mulStd = nn.SpatialConvolution(3,3,1,1):noBias()
        mulStd.weight = torch.Tensor({{std[1],0,0},{0,std[2],0},{0,0,std[3]}})

        model:insert(subMean,1)
        model:insert(divStd,2)
        model:add(mulStd)
        model:add(addMean)
    end

    return model
end

return createModel
