require 'nn'

--------------------------------------------------------------------------
-- Below are list of common modules used in various architectures.
-- Thoses are defined as global variables in order to make other codes uncluttered.
--------------------------------------------------------------------------

seq = nn.Sequential
conv = nn.SpatialConvolution
relu = nn.ReLU
bnorm = nn.SpatialBatchNormalization
shuffle = nn.PixelShuffle
deconv = nn.SpatialFullConvolution
pad = nn.Padding
concat = nn.ConcatTable
id = nn.Identity
cadd = nn.CAddTable

function addSkip(model)
    return seq()
        :add(concat()
            :add(model)
            :add(id()))
        :add(cadd(true))
end

function upsample(scale, method, nFeat)
    local scale = scale or 2
    local method = method or 'espcnn'
    local nFeat = nFeat or 64

    local model = seq()
    if method == 'deconv' then
        if scale == 2 then
            model:add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
            model:add(relu(true))
        elseif scale == 3 then
            model:add(deconv(nFeat,nFeat, 9,9, 3,3, 3,3))
            model:add(relu(true))
        elseif scale == 4 then
            model:add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
            model:add(relu(true))
            model:add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
            model:add(relu(true))
        end
    elseif method == 'espcnn' then  -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        if scale == 2 then
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(relu(true))
        elseif scale == 3 then
            model:add(conv(nFeat,9*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(3))
            model:add(relu(true))
        elseif scale == 4 then
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(relu(true))
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(relu(true))
        end
    end

    return model
end

function resBlock(nFeat, addBN, kernel)
    local nFeat = nFeat or 64
    local addBN = addBN or true
    local kernel = kernel or 3

    if addBN then
        return addSkip(seq()
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
            :add(relu(true))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat)))
    else
        return addSkip(seq()
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(relu(true))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1)))
    end
end

function cbrcb(nFeat)
    return seq()
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(bnorm(nFeat))
        :add(relu(true))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(bnorm(nFeat))
end

function brcbrc(nFeat)
    return seq()
        :add(bnorm(nFeat))
        :add(relu(true))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(bnorm(nFeat))
        :add(relu(true))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
end

function crc(nFeat)
    return seq()
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(relu(true))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
end