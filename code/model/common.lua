require 'nn'

--------------------------------------------------------------------------
-- Below are list of common modules used in various architectures.
-- Thoses are defined as global variables in order to make other codes uncluttered.
--------------------------------------------------------------------------

seq = nn.Sequential
conv = nn.SpatialConvolution
relu = nn.ReLU
prelu = nn.PReLU
rrelu = nn.RReLU
elu = nn.ELU
leakyrelu = nn.LeakyReLU
bnorm = nn.SpatialBatchNormalization
shuffle = nn.PixelShuffle
deconv = nn.SpatialFullConvolution
pad = nn.Padding
concat = nn.ConcatTable
id = nn.Identity
cadd = nn.CAddTable

function act(actParams, nOutputPlane)
    local nOutputPlane = actParams.nFeat or nOutputPlane
    local type = actParams.actType

    if type == 'relu' then
        return relu(true)
    elseif type == 'prelu' then
        return prelu(nOutputPlane)
    elseif type == 'rrelu' then
        return rrelu(actParams.l, actParams.u, true)
    elseif type == 'elu' then
        return elu(actParams.alpha, true)
    elseif type == 'leakyrelu' then
        return leakyrelu(actParams.negval, true)
    else
        error('unknown activation function!')
    end
end

function addSkip(model)
    return seq()
        :add(concat()
            :add(model)
            :add(id()))
        :add(cadd(true))
end

function upsample(scale, method, nFeat, actParams)
    local scale = scale or 2
    local method = method or 'espcnn'
    local nFeat = nFeat or 64

    local actType = actParams.actType
    local l, u = actParams.l, actParams.u
    local alpha, negval = actParams.alpha, actParams.negval
    actParams.nFeat = nFeat

    local model = seq()
    if method == 'deconv' then
        if scale == 2 then
            model:add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
            model:add(act(actParams))
        elseif scale == 3 then
            model:add(deconv(nFeat,nFeat, 9,9, 3,3, 3,3))
            model:add(act(actParams))
        elseif scale == 4 then
            model:add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
            model:add(act(actParams))
            model:add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
            model:add(act(actParams))
        end
    elseif method == 'espcnn' then  -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        if scale == 2 then
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(act(actParams))
        elseif scale == 3 then
            model:add(conv(nFeat,9*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(3))
            model:add(act(actParams))
        elseif scale == 4 then
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(act(actParams))
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(act(actParams))
        end
    end

    return model
end

function upsample_wo_act(scale, method, nFeat)
    local scale = scale or 2
    local method = method or 'espcnn'
    local nFeat = nFeat or 64

    if method == 'deconv' then
        if scale == 2 then
            return deconv(nFeat,nFeat, 6,6, 2,2, 2,2)
        elseif scale == 3 then
            return deconv(nFeat,nFeat, 9,9, 3,3, 3,3)
        elseif scale == 4 then
            return deconv(nFeat,nFeat, 6,6, 2,2, 2,2)
        end
    elseif method == 'espcnn' then  -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        if scale == 2 then
            return seq()
                :add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
                :add(shuffle(2))
        elseif scale == 3 then
            return seq()
                :add(conv(nFeat,9*nFeat, 3,3, 1,1, 1,1))
                :add(shuffle(3))
        elseif scale == 4 then
            return seq()
                :add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
                :add(shuffle(2))
                :add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
                :add(shuffle(2))
        end
    end
end

function resBlock(nFeat, addBN, actParams)
    local nFeat = nFeat or 64
    local addBN = addBN or true
    actParams.nFeat = nFeat

    if addBN then
        return addSkip(seq()
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
            :add(act(actParams))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat)))
    else
        return addSkip(seq()
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(act(actParams))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1)))
    end
end

function cbrcb(nFeat, actParams)
    return seq()
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(bnorm(nFeat))
        :add(act(actParams))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(bnorm(nFeat))
end

function brcbrc(nFeat, actParams)
    return seq()
        :add(bnorm(nFeat))
        :add(act(actParams))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(bnorm(nFeat))
        :add(act(actParams))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
end

function crc(nFeat, actParams)
    return seq()
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(act(actParams))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
end