require 'nn'

--nn.Criterion -> nn.CharbonnierCriterion
--Differentiable L1 loss function (supports batch)

--input:    input, target pairs (c x w x h) or (b x c x w x h)
--output:   A Charbonnier loss
--------------------------------------------------------------------------------
local CharbonnierCriterion, parent = torch.class('nn.CharbonnierCriterion', 'nn.Criterion')

function CharbonnierCriterion:__init(sizeAverage, eps)
    parent.__init(self)
    if (sizeAverage ~= nil) then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end

    if (eps ~= nil) then
        self.eps = eps * eps
    else
        self.eps = 0.001 * 0.001
    end

    self.buffer = nil
end

function CharbonnierCriterion:updateOutput(input, target)
    self.buffer = torch.sqrt(torch.pow(input - target, 2):add(self.eps))
    self.output = self.buffer:sum()
    if (self.sizeAverage) then
        self.output = self.output / input:nElement()
    end
    return self.output
end

function CharbonnierCriterion:updateGradInput(input, target)
    self.gradInput = (input - target):cdiv(self.buffer)
    if (self.sizeAverage) then
        self.gradInput = self.gradInput / input:nElement()
    end
    return self.gradInput
end


--test code
local input = torch.randn(5, 5)
local target = torch.randn(5, 5)
print(input)
print(target)

local cri_ref = nn.AbsCriterion()
local cri_imp = nn.CharbonnierCriterion()

print(cri_ref:forward(input, target))
print(cri_imp:forward(input, target))

print(cri_ref:backward(input, target))
print(cri_imp:backward(input, target))
--------------------------------------------------------------------------------

--[[
--test code 1(numerical differentiation)
local epsilon = 1e-5
local ref = cri_imp:forward(input, target)
local numerical = torch.zeros(5, 5)
for i = 1, 5 do
    for j = 1, 5 do
        local input_h = input:clone()
        input_h[i][j] = input_h[i][j] + epsilon
        numerical[i][j] = (cri_imp:forward(input_h, target) - ref) / epsilon
    end
end
print(numerical)
]]