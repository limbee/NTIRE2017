require('tvnorm-nn')

--------------------------------------------------------------------------------
local GradPriorCriterion, parent = torch.class('nn.GradPriorCriterion', 'nn.Criterion')

function GradPriorCriterion:__init(opt)
    parent.__init(self)
    
    self.criterion = nn.Sequential()
    self.criterion:add(nn.View(opt.batchSize * opt.nChannel, 1, opt.patchSize, opt.patchSize))
    self.criterion:add(nn.SpatialSimpleGradFilter())
    self.criterion:add(nn.Square())
    self.criterion:add(nn.Sum(2))
    self.criterion:add(nn.Power(opt.gradPower / 2))
    self.criterion:add(nn.SpatialAveragePooling(opt.patchSize - 1, opt.patchSize - 1))
    self.criterion:add(nn.View(opt.batchSize))
    self.criterion:add(nn.Mean())
    self.criterion = self.criterion:cuda()
    parent.cuda(self)
end

function GradPriorCriterion:updateOutput(input, target)
    self.output = self.criterion:forward(input)
    return self.output
end

function GradPriorCriterion:updateGradInput(input, target)
    self.gradInput = self.criterion:backward(input, torch.ones(1):cuda())
    return self.gradInput
end
--------------------------------------------------------------------------------
--[[
torch.setdefaulttensortype('torch.FloatTensor')
local opt = {batchSize = 4, nChannel = 1, patchSize = 4, patchSize = 4, gradPower = 1}

local ref = nn.SpatialTVNormCriterion():cuda()
local tes = nn.GradPriorCriterion(opt)

local input = torch.randn(4, 1, 4, 4):cuda()
local dummy = torch.randn(4, 1, 4, 4):cuda()
print('Input:')
print(input)

local oref = ref:forward(input, dummy)
local otes = tes:forward(input, dummy)

print('output_ref:')
print(oref)
print('output_tes:')
print(otes)

print('outputb_ref:')
print(ref:backward(input, dummy))
print('outputb_tes:')
print(tes:backward(input, dummy))
]]
