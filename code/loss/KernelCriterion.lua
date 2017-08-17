require 'nn'
require 'cunn'
require 'cudnn'

local SpatialConstConvolution, parent = torch.class('nn.SpatialConstConvolution', 'nn.Criterion')

function SpatialConstConvolution:__init(kernel)
    self.c = kernel:size(1)
    self.kW = kernel:size(2)
    self.kH = kernel:size(3)

    self.cConv = cudnn.SpatialConvolution(1, self.c, self.kW, self.kH, 1, 1, 0, 0)
    self.cConv.weight:copy(kernel)
    self.cConv:noBias()

    self.cConv = self.cConv:cuda()
    parent.cuda(self)
end

function SpatialConstConvolution:updateOutput(input)
    self.output = self.cConv:updateOutput(input)
    return self.output
end

function SpatialConstConvolution:updateGradInput(input, gradOutput)
    self.gradInput = self.cConv:updateGradInput(input, gradOutput)
    return self.gradInput
end

--------------------------------------------------------------------------------
local KernelCriterion, parent = torch.class('nn.KernelCriterion', 'nn.Criterion')

function KernelCriterion:__init(opt, kernel)
    parent.__init(self)
    
    self.inputGrad = nn.SpatialConstConvolution(kernel)
    self.targetGrad = nn.SpatialConstConvolution(kernel)
    self.gradDist = nil
    if opt.gradDist == 'mse' then
        self.gradDist = nn.MSECriterion(true)
    elseif opt.gradDist == 'abs' then
        self.gradDist = nn.AbsCriterion(true)
    end
    self.gradDist.sizeAverage = true

    parent.cuda(self)
end

function KernelCriterion:updateOutput(input, target)
    local b, c, h, w = unpack(input:size():totable())

    self.ig = self.inputGrad:forward(input:view(b * c, 1, h, w))
    self.tg = self.targetGrad:forward(target:view(b * c, 1, h, w))
    self.output = self.gradDist:forward(self.ig, self.tg)
    return self.output
end

function KernelCriterion:updateGradInput(input, target)
    local b, c, h, w = unpack(input:size():totable())
    
    self.dodg =  self.gradDist:updateGradInput(self.ig, self.tg)
    self.gradInput = self.inputGrad:updateGradInput(input:view(b * c, 1, h, w), self.dodg)
    self.gradInput  = self.gradInput:view(b, c, h, w)
    
    return self.gradInput
end
--------------------------------------------------------------------------------
--[[
local test = torch.randn(1, 4, 4):cuda()
local kernel = torch.CudaTensor(2, 2, 2)
print(kernel)
kernel[1][1][1] = -1
kernel[1][1][2] = 1
kernel[1][2][1] = 0
kernel[1][2][2] = 0
kernel[2][1][1] = -1
kernel[2][1][2] = 0
kernel[2][2][1] = 1
kernel[2][2][2] = 0
local ssgf = nn.SpatialSimpleGradFilter()
local tes = nn.SpatialConstConvolution(kernel)

print(ssgf:forward(test))
print(tes:forward(test))
]]
--[[
require 'image'
local img = image.load('butterfly.png'):cuda()
local c, h, w = table.unpack(img:size():totable())
img = img:view(c, 1, h, w)
local seq = nn.Sequential()
seq:add(nn.Square())
seq:add(nn.Sum(2))
seq:add(nn.Sqrt())
seq = seq:cuda()

local kernel1 = torch.CudaTensor({{{-1, 1}, {0, 0}}, {{-1, 0}, {1, 0}}})
local grad1 = nn.SpatialConstConvolution(kernel1)
local out1 = seq:forward(grad1:forward(img))
out1 = out1:view(c, h - 1, w - 1)
image.save('grad1.png', out1)

local kernel2 = torch.CudaTensor{{{0, 0, 0}, {1, -2, 1}, {0, 0, 0}}, {{0, 1, 0}, {0, -2, 0}, {0, 1, 0}}}
local grad2 = nn.SpatialConstConvolution(kernel2)
local out2 = seq:forward(grad2:forward(img))
out2 = out2:view(c, h - 2, w - 2)
image.save('grad2.png', out2)
]]
