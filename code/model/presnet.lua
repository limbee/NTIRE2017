require 'nn'
require 'model/common'

local MultiSkipAdd, parent = torch.class('nn.MultiSkipAdd', 'nn.Module')

function MultiSkipAdd:__init(ip)
    parent.__init(self)
    self.inplace = ip
end

--This function takes the input like {Skip, {Output1, Output2, ...}}
--and returns {Output1 + Skip, Output2 + Skip, ...}
--It also supports in-place calculation

function MultiSkipAdd:updateOutput(input)
    self.output = {}

    if self.inplace then
        for i = 1, #input[2] do
            self.output[i] = input[2][i]
        end
    else
        for i = 1, #input[2] do
            self.output[i] = input[2][i]:clone()
        end
    end

    for i = 1, #input[2] do
        self.output[i]:add(input[1])
    end
    
    return self.output
end

function MultiSkipAdd:updateGradInput(input, gradOutput)
    self.gradInput = {gradOutput[1]:clone():fill(0), {}}

    if self.inplace then
        for i = 1, #input[2] do
            self.gradInput[1]:add(gradOutput[i])
            self.gradInput[2][i] = gradOutput[i]
        end
    else
        for i = 1, #input[2] do
            self.gradInput[1]:add(gradOutput[i])
            self.gradInput[2][i] = gradOutput[i]:clone()
        end
    end

    return self.gradInput
end

local function createModel(opt)
    local addbn = opt.modelVer == 1
    local addrelu = (opt.modelVer == 1) or (opt.modelVer == 2)

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local model = seq()
        :add(conv(opt.nChannel, opt.nFeat, 3, 3, 1, 1, 1, 1))
    
    local nFront = 2 * opt.nResBlock / 3
    local nBack = opt.nResBlock - nFront

    local body = seq()
    for i = 1, nFront do
        body:add(resBlock(opt.nFeat, addbn, actParams))
    end

    local deep = seq()
    for i = 1, nBack do
        deep:add(resBlock(opt.nFeat, addbn, actParams))
    end

    body:add(concat()
        :add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))
        :add(deep
            :add(conv(opt.nFeat, opt.nFeat, 3, 3, 1, 1, 1, 1))))
    model:add(concat()
            :add(id())
            :add(body))
        :add(nn.MultiSkipAdd(true))
        :add(nn.ParallelTable()
            :add(seq()
                :add(upsample_wo_act(opt.scale, opt.upsample, opt.nFeat))
                :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1)))
            :add(seq()
                :add(upsample_wo_act(opt.scale, opt.upsample, opt.nFeat))
                :add(conv(opt.nFeat, opt.nChannel, 3, 3, 1, 1, 1, 1))))

    return model
end

return createModel

--this is a test code for nn.MultiSkipAdd
--[[
local conv = nn.SpatialConvolution(1, 1, 3, 3)
--local conv = nn.Identity()
local ref = nn.Sequential()
local cat = nn.ConcatTable()
local b1 = nn.Sequential()
    :add(nn.NarrowTable(1, 2))
    :add(nn.CAddTable(false))
    :add(conv:clone())
local b2 = nn.Sequential()
    :add(nn.ConcatTable()
        :add(nn.SelectTable(1))
        :add(nn.SelectTable(3)))
    :add(nn.CAddTable(false))
    :add(conv:clone())
ref
    :add(nn.FlattenTable())
    :add(cat
        :add(b1)
        :add(b2))

local comp = nn.Sequential()
    :add(nn.MultiSkipAdd(true))
    :add(nn.ParallelTable()
        :add(conv:clone())
        :add(conv:clone()))

local input1 = {torch.randn(1, 1, 6, 6), {torch.randn(1, 1, 6, 6), torch.randn(1, 1, 6, 6)}}
local input2 = {input1[1]:clone(), {input1[2][1]:clone(), input1[2][2]:clone()}}

print(input1)
print(table.unpack(ref:forward(input1)))
print(table.unpack(comp:forward(input2)))

local go1 = {torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4)}
local go2 = {go1[1]:clone(), go1[2]:clone()}

gi1 = ref:updateGradInput(input1, go1)
gi2 = comp:updateGradInput(input2, go2)
print(gi1[1])
print(table.unpack(gi1[2]))
print(gi2[1])
print(table.unpack(gi2[2]))
]]
