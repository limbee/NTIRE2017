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
    
    self.inputGrad = nn.Sequential()
    self.inputGrad:add(nn.View(opt.batchSize * opt.nChannel, 1, opt.patchSize, opt.patchSize))
    self.inputGrad:add(nn.SpatialConstConvolution(kernel))
    self.targetGrad = nn.Sequential()
    self.targetGrad:add(nn.View(opt.batchSize * opt.nChannel, 1, opt.patchSize, opt.patchSize))
    self.targetGrad:add(nn.SpatialConstConvolution(kernel))

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
    self.ig = self.inputGrad:forward(input)
    self.tg = self.targetGrad:forward(target)
    self.output = self.gradDist:forward(self.ig, self.tg)
    return self.output
end

function KernelCriterion:updateGradInput(input, target)
    self.dodg =  self.gradDist:updateGradInput(self.ig, self.tg)
    self.gradInput = self.inputGrad:updateGradInput(input, self.dodg)
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