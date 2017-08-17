require 'nn'
require 'cunn'

----------------------------------------------------------------------------
-- This file doesn't reproduce the original inception network (Szegedy 2015)
-- Our network was inspired by Inception-ResNet-v2, and brought some parts almost the same.
-- We don't use any of pooling or strided convolution here.
----------------------------------------------------------------------------

local function createModel(opt)
    local nFeat = opt.nFeat
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization
    local shuffle = nn.PixelShuffle
    local pad = nn.Padding
    local seq = nn.Sequential
    local concat = nn.ConcatTable
    local id = nn.Identity
    local cadd = nn.CAddTable
    local deconv = nn.SpatialFullConvolution
    local join = nn.JoinTable
    local mulc = nn.MulConstant

    local function cbr(nInputPlane, nOutputPlane, kH, kW, dH, dW, padH, padW)
        local kH, kW = kH or 3, kW or 3
        local dH, dW = dH or 1, dW or 1
        local padH, padW = padH or math.floor((kH - 1) / 2), padW or math.floor((kW - 1) / 2)
        return seq()
            :add(conv(nInputPlane, nOutputPlane, kH, kW, dH, dW, padH, padW))
            :add(bnorm(nFeat))
            :add(relu(true))
    end

    local function addSkip(layers)
        return seq()
            :add(concat()
                :add(layers)
                :add(id()))
            :add(cadd(true))
    end

    local function stem()
        return seq()
            :add(cbr(3, 32))
            :add(cbr(32, 32))
            :add(cbr(32, 64))
            :add(cbr(64, 96))
            :add(concat()
                :add(seq()
                    :add(cbr(96, 64, 1, 1))
                    :add(cbr(64, 96)))
                :add(seq()
                    :add(cbr(96, 64, 1, 1))
                    :add(cbr(64, 64, 7, 1))
                    :add(cbr(64, 64, 1, 7))
                    :add(cbr(64, 96))))
            :add(join(1, 3))
            :add(cbr(192, 192))
    end

    local function Inception_ResNet_A()
        return seq()
            :add(concat()
                :add(id())
                :add(seq()
                    :add(concat()
                        :add(seq()
                            :add(cbr(384, 32, 1, 1)))
                        :add(seq()
                            :add(cbr(384, 32, 1, 1))
                            :add(cbr(32, 32)))
                        :add(seq()
                            :add(cbr(384, 32, 1, 1))
                            :add(cbr(32, 48))
                            :add(cbr(48, 64))))
                    :add(join())
                    :add(conv(128, 384, 1, 1))
                    :add(bnorm(384))
                    :add(mulc(0.1, true))))
            :add(cadd())
            :add(relu(true))
    end

    local function Inception_ResNet_B()
        return seq()
            :add(concat(
                :add(id())
                :add(seq()
                    :add(concat()
                        :add(seq()
                            :add(1152, 192, 1, 1))
                        :add(seq()
                            :add(seq()
                                :add(cbr(1152, 128, 1, 1))
                                :add(cbr(128, 160, 1, 7))
                                :add(cbr(160, 192, 7, 1)))))
                    :add(join())
                    :add(conv(384, 1152, 1, 1))
                    :add(bnorm(1152))
                    :add(mulc(0.1, true)))))
            :add(cadd())
            :add(relu(true)))
    end

    local function Inception_ResNet_C()
        return seq()
            :add(concat(
                :add(id())
                :add(seq()
                    :add(concat()
                        :add(seq()
                            :add(2048, 192, 1, 1))
                        :add(seq()
                            :add(seq()
                                :add(cbr(2048, 192, 1, 1))
                                :add(cbr(192, 224, 1, 3))
                                :add(cbr(224, 256, 3, 1)))))
                    :add(join())
                    :add(conv(384, 1152, 1, 1))
                    :add(bnorm(1152))
                    :add(mulc(0.1, true)))))
            :add(cadd())
            :add(relu(true)))
    end



    return model
end

return createModel