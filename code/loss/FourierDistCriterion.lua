require 'nn'
require 'cunn'
require 'cudnn'

--for debugging
--require 'image'
--local signal = require 'signal'

--nn.Module -> nn.DFT2D
--Computes 2D DFT of input gray-scale image. (supports batch)

--input:    (1 x w x h) or (b x 1 x w x h)
--output:   {real (1 x w x h), image (1 x w x h)} 
--          or {real (b x 1 x w x h), image (b x 1 x w x h)}
--------------------------------------------------------------------------------
local DFT2D, parent = torch.class('nn.DFT2D', 'nn.Module')

function DFT2D:__init()
    parent.__init(self)
    
    self.b = -1
    self.w = -1
    self.h = -1
    self.rmm_1 = nn.MM(false, false):cuda()
    self.rmm_2_r = nn.MM(false, true):cuda()
    self.rmm_2_i = nn.MM(false, true):cuda()
    self.imm_1 = nn.MM(false, false):cuda()
    self.imm_2_r = nn.MM(false, true):cuda()
    self.imm_2_i = nn.MM(false, true):cuda()
    
    cudnn.convert(self.rmm_1, cudnn)
    cudnn.convert(self.rmm_2_r, cudnn)
    cudnn.convert(self.rmm_2_i, cudnn)
    cudnn.convert(self.imm_1, cudnn)
    cudnn.convert(self.imm_2_r, cudnn)
    cudnn.convert(self.imm_2_i, cudnn)
    parent.cuda(self)
end

--automatically sets kernel size
function DFT2D:_setSize(b, w, h)
    self.b = b
    self.w = w
    self.h = h
    self.sqn = math.sqrt(w * h)
    self.pre_real = torch.Tensor(w, w)
    self.pre_image = torch.Tensor(w, w)
    self.post_real = torch.Tensor(h, h)
    self.post_image = torch.Tensor(h, h)

    local basis = torch.ones(1, w, w) * (-2 * math.pi / w)
    for i = 1, w do
        for j = 1, w do
            basis[1][i][j] = basis[1][i][j] * (i - 1) * (j - 1)
        end
    end
    local basis_r = torch.expand(basis, b, w, w):clone()
    self.pre_real = torch.cos(basis_r):cuda()
    self.pre_image = torch.sin(basis_r):cuda()

    local basis = torch.ones(1, h, h) * (-2 * math.pi / h)
    for i = 1, h do
        for j = 1, h do
            basis[1][i][j] = basis[1][i][j] * (i - 1) * (j - 1)
        end
    end

    local basis_r = torch.expand(basis, b, h, h):clone()
    self.post_real = torch.cos(basis_r):cuda()
    self.post_image = torch.sin(basis_r):cuda()
end

function DFT2D:updateOutput(input)
    self.dim = 3
    if (input:dim() == 2) then
        input = input:view(1, self.w, self.h)
        self.dim = 2
    elseif (input:dim() == 4) then
        input = input:view(self.b, self.w, self.h)
        self.dim = 4
    end
    if ((self.b ~= input:size(1)) or (self.w ~= input:size(2)) or (self.h ~= input:size(3))) then
       self:_setSize(input:size(1), input:size(2), input:size(3)) 
    end

    self.rr_1 = self.rmm_1:forward({self.pre_real, input})
    self.ri_1 = self.imm_1:forward({self.pre_image, input})
    self.rr_2_1 = self.rmm_2_r:forward({self.rr_1, self.post_real})
    self.rr_2_2 = self.rmm_2_i:forward({self.ri_1, self.post_image})
    self.ri_2_1 = self.imm_2_r:forward({self.rr_1, self.post_image})
    self.ri_2_2 = self.imm_2_i:forward({self.ri_1, self.post_real})
    self.real = (self.rr_2_1 - self.rr_2_2)
    self.image = (self.ri_2_1 + self.ri_2_2)
    self.output = {self.real:div(self.sqn), self.image:div(self.sqn)}
    return self.output
end

function DFT2D:updateGradInput(input, gradOutput)
    if (self.dim == 2) then
        input = input:view(1, self.w, self.h)
    elseif (self.dim == 4) then
        input = input:view(self.b, self.w, self.h)
    end

    local grr_2_1 = self.rmm_2_r:updateGradInput({self.rr_1, self.post_real}, gradOutput[1])
    local grr_2_2 = self.rmm_2_i:updateGradInput({self.ri_1, self.post_image}, -gradOutput[1])
    local gri_2_1 = self.imm_2_r:updateGradInput({self.rr_1, self.post_image}, gradOutput[2])
    local gri_2_2 = self.imm_2_i:updateGradInput({self.ri_1, self.post_real}, gradOutput[2])
    local grr_1 = self.rmm_1:updateGradInput({self.pre_real, input}, grr_2_1[1] + gri_2_1[1])
    local gri_1 = self.imm_1:updateGradInput({self.pre_image, input}, grr_2_2[1] + gri_2_2[1])
    self.gradInput = (grr_1[2] + gri_1[2]):div(self.sqn)

    if (self.dim == 2) then
        self.gradInput = self.gradInput:view(self.w, self.h)
    elseif (self.dim == 4) then
        self.gradInput = self.gradInput:view(self.b, 1, self.w, self.h)
    end
    return self.gradInput
end
--------------------------------------------------------------------------------

--nn.Criterion -> nn.ComplexDistCriterion
--Computes distance between two complex tensors.

--params:   sizeAverage (true or false)
--input:    input, target pairs {real (b x w x h), image (b x w x h)}
--output:   A number
--------------------------------------------------------------------------------
local ComplexDistCriterion, parent = torch.class('nn.ComplexDistCriterion', 'nn.Criterion')

function ComplexDistCriterion:__init(sizeAverage)
    parent.__init(self)

    if (sizeAverage ~= nil) then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end

    parent.cuda(self)
end

function ComplexDistCriterion:updateOutput(input, target)
    
    local d = torch.pow(input[1] - target[1], 2) + torch.pow(input[2] - target[2], 2)
    self.output = d:sum()
    if (self.sizeAverage) then
        self.output = self.output / input[1]:nElement()
    end
    --This one is slow
    --[[local err_real = self.mseReal:forward(input[1], target[1])
    local err_image = self.mseImage:forward(input[2], target[2])
    self.output = err_real + err_image
    ]]
    return self.output
end

function ComplexDistCriterion:updateGradInput(input, target)

    self.gradInput = {(input[1] - target[1]):mul(2), (input[2] - target[2]):mul(2)}
    if (self.sizeAverage) then
        self.gradInput[1] = self.gradInput[1] / input[1]:nElement()
        self.gradInput[2] = self.gradInput[2] / input[2]:nElement()
    end
    --This one is slow
    --[[ local dedr = self.mseReal:backward(input[1], target[1])
    local dedi = self.mseImage:backward(input[2], target[2])
    self.gradInput = {dedr, dedi}
    ]]
    return self.gradInput
end
--------------------------------------------------------------------------------

--nn.Criterion -> nn.FourierDistCriterion
--Computes frequency-domain distance between two images. (supports batch)

--params:   sizeAverage (true or false)
--input:    input, target pairs (1 x w x h) or (b x 1 x w x h)
--output:   A number
--------------------------------------------------------------------------------
local FourierDistCriterion, parent = torch.class('nn.FourierDistCriterion', 'nn.Criterion')

function FourierDistCriterion:__init(sizeAverage)
    parent.__init(self)

    if (sizeAverage ~= nil) then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end

    self.transformI = nn.DFT2D()
    self.transformT = nn.DFT2D()
    self.criterion = nn.ComplexDistCriterion(sizeAverage)

    parent.cuda(self)
end

function FourierDistCriterion:updateOutput(input, target)
    self.Fi = self.transformI:forward(input)
    self.Ft = self.transformT:forward(target)
    self.output = self.criterion:forward(self.Fi, self.Ft)
    return self.output
end

function FourierDistCriterion:updateGradInput(input, target)
    local gradF = self.criterion:backward(self.Fi, self.Ft)
    self.gradInput = self.transformI:updateGradInput(input, gradF)
    return self.gradInput
end
--------------------------------------------------------------------------------

--nn.Criterion -> nn.FilteredDistCriterion
--Computes frequency-domain distance between two filtered images. (supports batch)
--You can make your own filter by modifying 'self.mask'.
--default: 'lpf' and 'hpf'

--params:   wc - cutoff frequency, filter - filter type, sizeAverage (true or false)
--input:    input, target pairs (1 x w x h) or (b x 1 x w x h)
--output:   A number
--------------------------------------------------------------------------------
local FilteredDistCriterion, parent = torch.class('nn.FilteredDistCriterion', 'nn.Criterion')

function FilteredDistCriterion:__init(wc, filter, sizeAverage)
    parent.__init(self)
    if (sizeAverage ~= nil) then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
    self.b = -1
    self.w = -1
    self.h = -1
    self.wc = wc
    self.filter = filter
    self.criterion = nn.ComplexDistCriterion(sizeAverage)
    
    parent.cuda(self)
end

function FilteredDistCriterion:_setMask(b, w, h)
    self.b = b
    self.w = w
    self.h = h
    self.transformI = nn.DFT2D()
    self.transformT = nn.DFT2D()
    self.wc_w = math.floor(w * self.wc)
    self.wc_h = math.floor(h * self.wc)
    self.lfMask_w = torch.ones(b, w, h)
    self.lfMask_h = torch.ones(b, w, h)
    self.lfMask_w[{{}, {self.wc_w + 1, w - self.wc_w}, {}}] = 0
    self.lfMask_h[{{}, {}, {self.wc_h + 1, h - self.wc_h}}] = 0
    self.mask = torch.cmul(self.lfMask_w, self.lfMask_h)
    if (self.filter == 'hpf') then
        self.mask = torch.ones(b, w, h) - self.mask
    elseif (self.filter == 'he') then
        local enhanceFactor = 5
        self.mask = (0.1 * torch.ones(b, w, h)) + (torch.ones(b, w, h) - self.mask)
    end
    self.mask = self.mask:cuda()
    --for debugging
    image.save('mask.png', self.mask)
end

function FilteredDistCriterion:updateOutput(input, target)
    self.dim = 3
    if (input:dim() == 2) then
        input = input:view(1, self.w, self.h)
        target = target:view(1, self.w, self.h)
        self.dim = 2
    elseif (input:dim() == 4) then
        input = input:view(self.b, self.w, self.h)
        target = target:view(self.b, self.w, self.h)
        self.dim = 4
    end

    if ((self.b ~= input:size(1)) or (self.w ~= input:size(2)) or (self.h ~= input:size(3))) then
       self:_setMask(input:size(1), input:size(2), input:size(3)) 
    end

    self.Fi = self.transformI:forward(input)
    self.Fi[1]:cmul(self.mask)
    self.Fi[2]:cmul(self.mask)
    self.Ft = self.transformT:forward(target)
    self.Ft[1]:cmul(self.mask)
    self.Ft[2]:cmul(self.mask)
    self.output = self.criterion:forward(self.Fi, self.Ft)
    return self.output
end

function FilteredDistCriterion:updateGradInput(input, target)
    local gradF = self.criterion:backward(self.Fi, self.Ft)
    self.gradInput = self.transformI:updateGradInput(input, gradF)
    if (self.dim == 2) then
        self.gradInput = self.gradInput:view(self.w, self.h)
    elseif (self.dim == 4) then
        self.gradInput = self.gradInput:view(self.b, 1, self.w, self.h)
    end
    return self.gradInput
end
--------------------------------------------------------------------------------
--[[
--test code 1
local b = image.load('gray.png'):cuda()
local w = b:size(2)
local h = b:size(3)
local a = torch.zeros(1, w, h):cuda()

local mod = nn.DFT2D(1, w, h)
--local cri = nn.FourierDistCriterion(1, w, h):cuda()
local cri = nn.FilteredDistCriterion(0.2, 'he'):cuda()

--print(unpack(mod:forward(a)))

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
image.save('FD_opt.png', oa)
--print(oa)
--print(b)
]]
--[[
--test code 2
print('a:')
print(a)
print('b:')
print(b)
print('a - b:')
print(a - b)

print('MSE loss')
local mse = nn.MSECriterion():cuda()
local mseLoss = mse:forward(a, b)
local dmseda = mse:backward(a, b)
print(mseLoss)
print(dmseda)
print('Fourier loss')
local ref = cri:forward(a, b)
local da = cri:backward(a, b)
print(ref)
print(da)
local numerical = torch.zeros(4, 4)
local epsilon = 1e-4

for i = 1, 4 do
    for j = 1, 4 do
        local a_h = a:clone()
        a_h[i][j] = a_h[i][j] + epsilon
        numerical[i][j] = (cri:forward(a_h, b) - ref) / epsilon
    end
end
print(numerical)
]]