require('tvnorm-nn')

--------------------------------------------------------------------------------
local GradCriterion, parent = torch.class('nn.GradCriterion', 'nn.Criterion')

function GradCriterion:__init(opt)
    parent.__init(self)

    self.inputGrad = nn.Sequential()
    self.inputGrad:add(nn.View(opt.batchSize * opt.nChannel, 1, opt.patchSize, opt.patchSize))
    self.inputGrad:add(nn.SpatialSimpleGradFilter())
    self.targetGrad = nn.Sequential()
    self.targetGrad:add(nn.View(opt.batchSize * opt.nChannel, 1, opt.patchSize, opt.patchSize))
    self.targetGrad:add(nn.SpatialSimpleGradFilter())

    self.gradDist = nil
    if opt.gradDist == 'mse' then
        self.gradDist = nn.MSECriterion(true)
    elseif opt.gradDist == 'abs' then
        self.gradDist = nn.AbsCriterion(true)
    end
    self.gradDist.sizeAverage = true
--[[
    self.wGradPrior = opt.gradPrior
    self.gradPrior = nn.Sequential()
    self.gradPrior:add(nn.View(opt.batchSize * opt.nChannel, 1, opt.patchSize, opt.patchSize))
    self.gradPrior:add(nn.SpatialSimpleGradFilter())
    self.gradPrior:add(nn.Square())
    self.gradPrior:add(nn.Sum(2))
    self.gradPrior:add(nn.Pow(0.4))
    self.gradPrior:add(nn.Mean())
]]
    parent.cuda(self)
end

function GradCriterion:updateOutput(input, target)
    --[[self.iTemp = self.inputGrad:forward(input)
    self.tTemp = self.targetGrad:forward(target)
    self.output = self.gradDist:forward(self.iTemp, self.tTemp) + (wGradPrior * self.gradPrior:forward(input))]]
    self.ig = self.inputGrad:forward(input)
    self.tg = self.targetGrad:forward(target)
    self.output = self.gradDist:forward(self.ig, self.tg)
    return self.output
end

function GradCriterion:updateGradInput(input, target)
    self.dodg =  self.gradDist:updateGradInput(self.ig, self.tg)
    self.gradInput = self.inputGrad:updateGradInput(input, self.dodg)
    return self.gradInput
end
--------------------------------------------------------------------------------
