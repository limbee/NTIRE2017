require 'nn'
require 'cunn'
require 'cudnn'

--for debugging
--require 'image'

--nn.Criterion -> nn.SSIMCriterion
--Computes SSIM between two images (supports batch)

--input:    input, target pairs (c x w x h) or (b x c x w x h)
--output:   A number (SSIM)
--------------------------------------------------------------------------------
local SSIMCriterion, parent = torch.class('nn.SSIMCriterion', 'nn.Criterion')

--nn.Criterion -> nn.SSIMCriterion
--Computes SSIM between two images (supports batch)

--input:    input, target pairs (c x w x h) or (b x c x w x h)
--output:   A number (SSIM)
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

    parent.cuda(self)
end

function SSIMCriterion:updateOutput(input, target)
    if (input:dim() ~= 4) then
        self.mx = input:mean()
        self.my = target:mean()
        self.vx = torch.pow(input, 2):mean() - (self.mx * self.mx)
        self.vy = torch.pow(target, 2):mean() - (self.my * self.my)
        self.cov = torch.cmul(input, target):mean() - (self.mx * self.my)
        
        self.mn = (2 * self.mx * self.my) + self.c1
        self.md = self.mx^2 + self.my^2 + self.c1
        self.meanTerm = self.mn / self.md

        self.vn = (2 * self.cov) + self.c2
        self.vd = (self.vx + self.vy) + self.c2
        self.varTerm = self.vn / self.vd

        self.output = self.meanTerm * self.varTerm
    else
        self.b = input:size(1)
        self.c = input:size(2)
        self.w = input:size(3)
        self.h = input:size(4)
        self.bi = input:view(self.b, -1)
        self.bt = target:view(self.b, -1)

        self.mx = self.bi:mean(2):view(-1)
        self.my = self.bt:mean(2):view(-1)
        self.vx = (torch.pow(self.bi, 2):mean(2) - torch.cmul(self.mx, self.mx)):view(-1)
        self.vy = (torch.pow(self.bt, 2):mean(2) - torch.cmul(self.my, self.my)):view(-1)
        self.cov = (torch.cmul(self.bi, self.bt):mean(2) - torch.cmul(self.mx, self.my)):view(-1)
        
        self.mn = (2 * torch.cmul(self.mx, self.my)) + self.c1
        self.md = (torch.pow(self.mx, 2) + torch.pow(self.my, 2)) + self.c1
        self.meanTerm = torch.cdiv(self.mn, self.md)

        self.vn = (2 * self.cov) + self.c2
        self.vd = (self.vx + self.vy) + self.c2
        self.varTerm = torch.cdiv(self.vn, self.vd)

        self.output = torch.cmul(self.meanTerm, self.varTerm):mean()
    end
    
    return self.output
end

function SSIMCriterion:updateGradInput(input, target)
    if (input:dim() ~= 4) then
        local ne = input:nElement()
        local dMdmu = 2 * ((self.my * self.md) - (self.mx * self.mn)) / self.md^2
        local dMdx = torch.CudaTensor(input:size()):fill(dMdmu / ne)

        local dVdvar = -self.vn / self.vd^2
        local dvardx = (input - self.mx) * 2 / ne
        local dVdcov = 2 / self.vd
        local dcovdx = (target - self.my) / ne
        local dVdx = (dvardx * dVdvar) + (dcovdx * dVdcov)

        self.gradInput = -(dMdx * self.varTerm) - (dVdx * self.meanTerm)
    else
        local ne = self.bi[1]:nElement()
        local dMdmu = 2 * (torch.cmul(self.my, self.md) - torch.cmul(self.mx, self.mn))
        dMdmu = torch.cdiv(dMdmu, torch.pow(self.md, 2))
        local dMdx = torch.repeatTensor(dMdmu:view(self.b, 1, 1, 1), 1, self.c, self.w, self.h):div(ne)
        
	    local dVdvar = -torch.cdiv(self.vn, torch.pow(self.vd, 2))
	    local mxe = torch.repeatTensor(self.mx:view(self.b, 1, 1, 1), 1, self.c, self.w, self.h)
        local mye = torch.repeatTensor(self.my:view(self.b, 1, 1, 1), 1, self.c, self.w, self.h)
        local dvardx = (input - mxe) * 2 / ne
        local dVdcov = torch.cdiv(torch.CudaTensor(self.b):fill(2), self.vd)
        local dcovdx = (target - mye) / ne
        local dVdvare = torch.repeatTensor(dVdvar:view(self.b, 1, 1, 1), 1, self.c, self.w, self.h)
        local dVdcove = torch.repeatTensor(dVdcov:view(self.b, 1, 1, 1), 1, self.c, self.w, self.h)
        local dVdx = torch.cmul(dvardx, dVdvare) + torch.cmul(dcovdx, dVdcove)

        local meanTerme = torch.repeatTensor(self.meanTerm:view(self.b, 1, 1, 1), 1, self.c, self.w, self.h)
        local varTerme = torch.repeatTensor(self.varTerm:view(self.b, 1, 1, 1), 1, self.c, self.w, self.h)
        
        self.gradInput = -torch.cmul(dMdx, varTerme) - torch.cmul(dVdx, meanTerme)
    end

    return self.gradInput
end
--------------------------------------------------------------------------------

--[[
--test code 1 (numerical differentiation)
local input = torch.randn(5, 5):div(10):add(0.5):clamp(0, 1)
local target = torch.randn(5, 5):div(10):add(0.5):clamp(0, 1)
print('Input:')
print(input)
print('Target:')
print(target)

print('SSIM:')
local cri_SSIM = nn.SSIMCriterion()
print(cri_SSIM:forward(input, target))

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
--[[
b = image.load('gray.png'):cuda()
local w = b:size(2)
local h = b:size(3)
local a = torch.randn(1, w, h):cuda()

local cri = nn.SSIMCriterion():cuda()

--[[
--test code 2 (optimization)
b = image.load('gray.png'):cuda()
local w = b:size(2)
local h = b:size(3)
local a = torch.randn(1, w, h):cuda()

local cri = nn.SSIMCriterion():cuda()

local oa = a:clone()
local lr = 200
for i = 1, 1000 do
    local err = cri:forward(oa, b)
    if ((i % 50) == 0) then
        print(err)
    end
    local da = cri:backward(oa, b)
    oa = oa - da:mul(lr)
end
image.save('SSIM_opt.png', oa)
]]

--[[
--test code 3 (batch)
ba = torch.randn(4, 3, 8, 8):cuda()
bb = torch.randn(4, 3, 8, 8):cuda()
cri = nn.SSIMCriterion()
print(cri:forward(ba, bb))
print(cri:backward(ba, bb))
a = ba[1]
b = bb[1]
print(cri:forward(a, b))
print(cri:backward(a, b))
]]