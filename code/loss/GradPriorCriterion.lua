require('tvnorm-nn')

--------------------------------------------------------------------------------
local GradPriorCriterion, parent = torch.class('nn.GradPriorCriterion', 'nn.Criterion')

function GradPriorCriterion:__init(opt)
    parent.__init(self)

    self:add(nn.Sequential())
    self:add(nn.View(opt.batchSize * opt.nChannel, 1, opt.patchSize, opt.patchSize))
    self:add(nn.SpatialSimpleGradFilter())
    self:add(nn.Square())
    self:add(nn.Sum(2))
    self:add(nn.Pow(opt.gradPower / 2))
    self:add(nn.Mean())

    parent.cuda(self)
end

function GradPriorCriterion:updateOutput(input, target)
    return parent.updateOutput(self, input)
end

function GradPriorCriterion:updateGradInput(input, target)
    return parent.updateGradInput(self, input, Torch.ones(1):cuda())
end
--------------------------------------------------------------------------------
--[[
local opt = {batchSize = 4, nChannel = 1, patchSize = 4, patchSize = 4, gradPower = 1}

local ref = nn.SpatialTVNormCriterion()
local tes = nn.GradPriorCriterion()

local input = torch.randn(4, 1, 4, 4):cuda()
local dummy = torch.randn(4, 1, 4, 4):cuda()
print(input)

local or = ref:forward(input, dummy)
local ot = tes:forward(input, dummy)

print(or)
print(ot)

print(ref:backward(input, dummy))
print(tes:backward(input, dummy))
]]