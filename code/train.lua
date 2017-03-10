local image = require 'image'
local optim = require 'optim'
local util = require 'utils'()

local M = {}
local Trainer = torch.class('sr.Trainer', M)

function Trainer:__init(model, criterion, opt)
    self.model = model
    self.criterion = criterion
    self.opt = opt
    self.optimState = opt.optimState

    self.params, self.gradParams = model:getParameters()
    self.feval = function() return self.err, self.gradParams end
end

function Trainer:train(epoch, dataloader)
    local size = dataloader:size()
    local trainTimer = torch.Timer()
    local dataTimer = torch.Timer()
    local trainTime, dataTime = 0, 0
    local iter, err = 0, 0

    cudnn.fastest = true
    cudnn.benchmark = true

    self.model:training()
    for n, sample in dataloader:run() do
        dataTime = dataTime + dataTimer:time().real
        
        --Copy input and target to the GPU
        self:copyInputs(sample, 'train')
        sample = nil
        collectgarbage()
        collectgarbage()

        self.model:zeroGradParameters()
        self.model:forward(self.input)
        err = err + self.criterion(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)
        if self.opt.clip > 0 then
            self.gradParams:clamp(-self.opt.clip / self.opt.lr, self.opt.clip / self.opt.lr)
        end
        self.optimState.method(self.feval, self.params, self.optimState)
        trainTime = trainTime + trainTimer:time().real

        iter = iter + 1
        if n % self.opt.printEvery == 0 then
            local it = (epoch - 1) * self.opt.testEvery + n
            print(('[Iter: %.1fk] Time: %.2f (data: %.2f),\terr: %.6f')
                :format(it / 1000, trainTime, dataTime, err/iter))
            err, iter = 0, 0
            trainTime, dataTime = 0, 0
        end

        if n % self.opt.testEvery == 0 then
            break
        end

        trainTimer:reset()
        dataTimer:reset()
    end
    
    return err / iter
end

function Trainer:test(epoch, dataloader)
    local timer = torch.Timer()
    local iter, avgPSNR = 0, 0
    local mean = self.opt.mean:clone():cuda()
    local std = self.opt.std:clone():cuda()

    self.model:clearState()
    self.model:evaluate()
    collectgarbage()
    collectgarbage()

    cudnn.fastest = false
    cudnn.benchmark = false
    
    for n, sample in dataloader:run() do
        self:copyInputs(sample,'test')
        sample = nil
        collectgarbage()
        collectgarbage()

        local input = nn.Unsqueeze(1):cuda():forward(self.input)
        if self.opt.nChannel == 1 then
            input = nn.Unsqueeze(1):cuda():forward(input)
        end
        local outputFull = util:recursiveForward(input, self.model)
        if self.opt.netType == 'bandnet' then
            output = outputFull[2]:squeeze(1)
        else
            output = outputFull:squeeze(1)
        end

        if self.opt.subMean then
            local h,w = output:size(2), output:size(3)
            output:add(1, mean:repeatTensor(1,h,w))
            self.target:add(1, mean:repeatTensor(1,h,w))
            if self.opt.divStd then
                output:cmul(std:repeatTensor(1,h,w))
                self.target:cmul(std:repeatTensor(1,h,w))
            end
        end

        avgPSNR = avgPSNR + util:calcPSNR(output, self.target, self.opt.scale)
        image.save(paths.concat(self.opt.save, 'result', n .. '.png'), output:float():squeeze())

        if self.opt.netType == 'bandnet' then
            local outputLow = outputFull[1][1]:squeeze(1):div(255)
            local outputHigh = outputFull[1][2]:squeeze(1):div(255)
            image.save(paths.concat(self.opt.save, 'result', n .. '_low.png'), outputLow)
            image.save(paths.concat(self.opt.save, 'result', n .. '_high.png'), outputHigh)
        end
        iter = iter + 1
        self.model:clearState()
        output = nil
        outputFull = nil
        collectgarbage()
        collectgarbage()
    end

    print(('[epoch %d (iter/epoch: %d)] Average PSNR: %.4f,  Test time: %.2f\n')
        :format(epoch, self.opt.testEvery, avgPSNR / iter, timer:time().real))

    return avgPSNR / iter
end

function Trainer:copyInputs(sample, mode)
    if mode == 'train' then
        self.input = self.input or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
    elseif mode == 'test' then
        self.input = self.input or torch.CudaTensor()
    end

    self.input:resize(sample.input:size()):copy(sample.input)
    self.target = self.target or torch.CudaTensor()
    self.target:resize(sample.target:size()):copy(sample.target)

    sample = nil
    collectgarbage()
    collectgarbage()
end

return M.Trainer
