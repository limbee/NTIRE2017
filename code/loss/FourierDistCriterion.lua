require 'nn'
require 'cunn'
require 'cudnn'

--for debugging
--require 'image'
--local signal = require 'signal'

--nn.Module -> nn.DFT2D
--Computes 2D DFT of input gray-scale image. (supports batch)

--input:    (b x 1 x w x h) or (b x 3 x w x h)
--output:   {real (b x 1 x w x h), image (b x 1 x w x h)} 
--          or {real (b x 3 x w x h), image (b x 3 x w x h)}
--------------------------------------------------------------------------------
local DFT2D, parent = torch.class('nn.DFT2D', 'nn.Module')

function DFT2D:__init()
    parent.__init(self)
    
    self.sz = torch.CudaTensor({-1, -1, -1, -1})
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
function DFT2D:_setSize(sz)
    self.sz = sz:clone()
    local b, c, w, h = self.sz[1], self.sz[2], self.sz[3], self.sz[4]
    self.sqn = math.sqrt(w * h)

    local cv_pre = torch.linspace(0, w - 1, w):repeatTensor(w, 1)
    local basis_pre = torch.cmul(cv_pre, cv_pre:t()):mul(-2 * math.pi / w)
    local basis_pre = torch.expand(basis_pre:view(1, w, w), b, w, w)
    self.pre_real = torch.cos(basis_pre):cuda()
    self.pre_image = torch.sin(basis_pre):cuda()

    local cv_post = torch.linspace(0, h - 1, h):repeatTensor(h, 1)
    local basis_post = torch.cmul(cv, cv:t()):mul(-2 * math.pi / h)
    local basis_post = torch.expand(basis_post:view(1, h, h), b, h, h)
    self.post_real = torch.cos(basis_post):cuda()
    self.post_image = torch.sin(basis_post):cuda()
end

function DFT2D:updateOutput(input)
    if (torch.all(torch.eq(input:size(), self.sz)) == false) then
        self:_setSize(input:size())
    end
    input:view(input, self.sz[1] * self.sz[2], self.sz[3], self.sz[4])
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
    local grr_2_1 = self.rmm_2_r:updateGradInput({self.rr_1, self.post_real}, gradOutput[1])
    local grr_2_2 = self.rmm_2_i:updateGradInput({self.ri_1, self.post_image}, -gradOutput[1])
    local gri_2_1 = self.imm_2_r:updateGradInput({self.rr_1, self.post_image}, gradOutput[2])
    local gri_2_2 = self.imm_2_i:updateGradInput({self.ri_1, self.post_real}, gradOutput[2])
    local grr_1 = self.rmm_1:updateGradInput({self.pre_real, input}, grr_2_1[1] + gri_2_1[1])
    local gri_1 = self.imm_1:updateGradInput({self.pre_image, input}, grr_2_2[1] + gri_2_2[1])
    self.gradInput = (grr_1[2] + gri_1[2]):div(self.sqn)
    self.gradInput:view(self.gradInput, self.sz[1], self.sz[2], self.sz[3], self.sz[4])

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
--This code is commented because it is same with MSECriterion
--------------------------------------------------------------------------------
--[[
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
]]
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
    self.transformI = nn.DFT2D()
    self.transformT = nn.DFT2D()

    self.sz = torch.CudaTensor({-1, -1, -1, -1})
    self.wc = wc
    self.filter = filter
    self.criterion = nn.ComplexDistCriterion(sizeAverage)
    
    parent.cuda(self)
end

function FilteredDistCriterion:_makeLMask(wc)
    local b, c, w, h = self.sz[1], self.sz[2], self.sz[3], self.sz[4]
    local wc_w = math.floor(w * wc)
    local wc_h = math.floor(h * wc)
    local lfMask_w = torch.ones(b, w, h)
    local lfMask_h = torch.ones(b, w, h)
    lfMask_w[{{}, {wc_w + 1, w - wc_w}, {}}] = 0
    lfMask_h[{{}, {}, {wc_h + 1, h - wc_h}}] = 0

    return torch.cmul(lfMask_w, lfMask_h)
end

function FilteredDistCriterion:_setMask(sz)
    self.sz = sz:clone()
    local b, c, w, h = self.sz[1], self.sz[2], self.sz[3], self.sz[4]
    self.mask = self:_makeLMask(self.wc)
    local eF = 1
    if (self.filter == 'highpass') then
        self.mask = torch.ones(b, w, h) - self.mask
    elseif (self.filter == 'high_enhance') then
        eF = 3
        local hfMask = torch.ones(b, w, h) - self.mask
        self.mask = torch.ones(b, w, h) + hfMask:mul(eF - 1)
    end
    self.mask = self.mask:cuda()
    --for debugging
    image.save('mask.png', self.mask / eF)
end

function FilteredDistCriterion:updateOutput(input, target)
    if (torch.all(torch.eq(input:size(), self.sz)) == false) then
        self:_setMask(input:size())
    end
    self.Fi = self.transformI:forward(input)
    self.Ft = self.transformT:forward(target)
    self.Fi[1]:cmul(self.mask)
    self.Fi[2]:cmul(self.mask)
    self.Ft[1]:cmul(self.mask)
    self.Ft[2]:cmul(self.mask)
    self.output = self.criterion:forward(self.Fi, self.Ft)

    return self.output
end

function FilteredDistCriterion:updateGradInput(input, target)
    local gradF = self.criterion:backward(self.Fi, self.Ft)
    self.gradInput = self.transformI:updateGradInput(input, gradF)
    self.gradInput:view(self.gradInput, self.sz[1], self.sz[2], self.sz[3], self.sz[4])

    return self.gradInput
end
--------------------------------------------------------------------------------

--test code 1
local b = image.load('color.png'):cuda()
local c = b:size(1)
local w = b:size(2)
local h = b:size(3)
b = b:view(1, c, w, h)
local a = torch.ones(1, c, w, h):clamp(0, 1):cuda()

local mod = nn.DFT2D()
local crifd = nn.FilteredDistCriterion(0.5, 'lowpass'):cuda()
local crimse = nn.MSECriterion()
local cri = nn.MultiCriterion()
cri:add(crifd, 1)
cri:add(crimse, 0.1)
cri:cuda()
--print(unpack(mod:forward(a)))

local oa = a:clone()
local lr = 200
for i = 1, 3000 do
    local err = cri:forward(oa, b)
    if ((i % 300) == 0) then
        print(err)
    end
    local da = cri:backward(oa, b)
    oa = oa - da:mul(lr)
end
oa = oa:squeeze(1)
image.save('FD_opt.png', oa)
image.save('FD_gray.png', image.rgb2yuv(oa)[1])

--[[
--test code 2
a = torch.Tensor({{1, 2, 3, 4}, {4, 5, 6, 5}, {7, 8, 9, 8}, {4, 3, 2, 1}}):cuda()
b = torch.Tensor({{1, 2, 1, 0}, {4, 3, 2, 1}, {6, 4, 1, 3}, {4, 2, 3, 1}}):cuda()
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