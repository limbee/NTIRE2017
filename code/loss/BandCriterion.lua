require('loss/FourierDistCriterion')

--------------------------------------------------------------------------------
local BandCriterion, parent = torch.class('nn.BandCriterion', 'nn.Criterion')

function BandCriterion:__init(wc, lrLow, lrHigh, sizeAverage)
    parent.__init(self)

    self.lrLow = lrLow
    self.lrHigh = lrHigh
    self.lowBand = nn.FilteredDistCriterion(wc / 2, 'lowpass', sizeAverage)
    self.highBand = nn.FilteredDistCriterion(wc / 2, 'highpass', sizeAverage)
    self.total = nn.MSECriterion():cuda()
    self.total.sizeAverage = sizeAverage

    parent.cuda(self)
end

function BandCriterion:updateOutput(input, target)
    local errLow = self.lowBand:forward(input[1][1], target)
    local errHigh = self.highBand:forward(input[1][2], target)
    local errTotal = self.total:forward(input[2], target)
    self.output = (errLow + errHigh + errTotal) / 3

    return self.output
end

function BandCriterion:updateGradInput(input, target)
    local gradLow = self.lowBand:updateGradInput(input[1][1], target)
    local gradHigh = self.highBand:updateGradInput(input[1][2], target)
    local gradTotal = self.total:updateGradInput(input[2], target)
    self.gradInput =
    {
        {self.lrLow * gradLow / 3, self.lrHigh * gradHigh / 3},
        gradTotal / 3
    }
    
    return self.gradInput
end
--------------------------------------------------------------------------------