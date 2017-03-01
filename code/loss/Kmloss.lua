local KMLCriterion, parent = torch.class('nn.KMLCriterion', 'nn.Criterion')

function KMLCriterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = false
   end
    self.effectLossLowerBound = 0.5

end

function KMLCriterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.MSECriterion_updateOutput(
      input:cdata(),
      target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage
   )
   self.output = self.output_tensor[1]

  errorCut = torch.round(torch.cmin(torch.ones(target:size()):float(),torch.abs(target-input):float())-self.effectLossLowerBound)
   self.output = self.output/torch.sum(errorCut)
   print(self.output)
   return self.output
end

function KMLCriterion:updateGradInput(input, target)
   input.THNN.MSECriterion_updateGradInput(
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage
   )
   errorCut = torch.round(torch.cmin(torch.ones(target:size()):float(),torch.abs(target-input):float())-self.effectLossLowerBound)
   self.gradInput = self.gradInput/torch.sum(errorCut)
   print(torch.sum(self.gradInput))
   return self.gradInput
end