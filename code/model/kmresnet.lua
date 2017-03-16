require 'nn'
require 'cunn'
-- Kyoung Mu's SRResNet
-- input: ILR (bicubic interpolated LR)
local function createModel(opt)

    local fltsz = opt.filtsize
    local padsz = (fltsz-1)/2
    local nBlocks = opt.nResBlock
    local nFeat = opt.nFeat
    local nChannel = opt.nChannel
    
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local seq = nn.Sequential
    local concat = nn.ConcatTable
    local id = nn.Identity
    local cadd = nn.CAddTable
    
    assert(opt.dataSize == 'big', "KyoungMu's SRResNet requires bicubic interpolation!")

    local function convBlock(nFeat)
        local s = nn.Sequential()
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(relu(true))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        return seq()
            :add(concat()
                :add(s)
                :add(id()))
            :add(cadd(true))
    end

    local model = seq()
        :add(bias(-opt.mulImg/2, true))
        :add(conv(nChannel,nFeat, fltsz,fltsz, 1,1, padsz,padsz))
    for i = 1, opt.nResBlock do
        local block = convBlock(nFeat)
        for j = 1, block:size() do
            model:add(block:get(j):clone())
        end
        block = nil
    end
    model:add(conv(nFeat,nChannel, fltsz,fltsz, 1,1, padsz,padsz))
    model:add(bias(opt.mulImg/2, true))

	model:reset()
    
    return model
end

return createModel