require 'nn'

local SSIMCriterion, parent = torch.class('nn.SSIMCriterion', 'nn.Criterion')

function SSIMCriterion:__init(params)
    parent.__init(self)

    self.mx = nil
    self.my = nil
    self.vx = nil
    self.vy = nil
    self.cov = nil

    if (params == nil) then
        self.L = 1
        self.k1 = 0.01
        self.k2 = 0.03
    else
        assert(#params == 3)
        self.L = params[1]
        self.k1 = params[2]
        self.k2 = params[3]
    end

	self.c1 = (self.k1 * self.L)^2
    self.c2 = (self.k2 * self.L)^2
end

function SSIMCriterion:updateOutput(input, target)
    self.mx = input:mean()
    self.my = target:mean()
    self.vx = torch.pow(input, 2):mean() - (self.mx * self.mx)
    self.vy = torch.pow(target, 2):mean() - (self.my * self.my)
    self.cov = torch.cmul(input, target):mean() - (self.mx * self.my)
    
    self.mn = (2 * self.mx * self.my) + self.c1
    self.md = self.mx^2 + self.my^2 + self.c1
    self.meanTerm = self.mn / self.md

    self.vn = (2 * self.cov) + self.c2
    self.vd = self.vx + self.vy + self.c2
    self.varTerm = self.vn / self.vd

    self.output = self.meanTerm * self.varTerm
    return self.output
end

function SSIMCriterion:updateGradInput(input, target)
    local ne = input:nElement()
    local dMdmu = ((2 * self.my * self.md) - (2 * self.mx * self.mn)) / self.md^2
    local dMdx = torch.ones(input:size()):mul(dMdmu / ne)

    local dVdvar = -self.vn / self.vd^2
    local dvardx = (input - self.mx):mul(2 / ne)
    local dVdcov = 2 / self.vd
    local dcovdx = (target - self.my):div(ne)
    local dVdx = dvardx:mul(dVdvar) + dcovdx:mul(dVdcov)

    self.gradInput = -dMdx:mul(self.varTerm) - dVdx:mul(self.meanTerm)
    return self.gradInput
end

--test code
local input = torch.randn(5, 5):div(10):add(0.5):clamp(0, 1)
local target = torch.randn(5, 5):div(10):add(0.5):clamp(0, 1)
print('Input:')
print(input)
print('Target:')
print(target)

print('SSIM:')
local cri_SSIM = nn.SSIMCriterion()
print(cri_SSIM:forward(input, target))

--numerical differentiation
--[[
local epsilon = 1e-5
local ref = cri_SSIM:forward(input, target)
local numerical = torch.zeros(5, 5)
for i = 1, 5 do
    for j = 1, 5 do
        local input_h = input:clone()
        input_h[i][j] = input_h[i][j] + epsilon
        numerical[i][j] = (cri_SSIM:forward(input_h, target) - ref) / epsilon
    end
end
print(numerical)
]]

local input_o = input:clone()
local lr = 0.1
for i = 1, 100 do
    local updateSSIM = cri_SSIM:forward(input_o, target)
    if ((i % 100) == 0) then
        print(updateSSIM)
    end
    input_o = input_o - (lr * cri_SSIM:backward(input_o, target))
end
print('Result:')
print(input_o)