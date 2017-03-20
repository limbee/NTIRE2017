local image = require 'image'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('sr.Trainer', M)

function Trainer:__init(model, criterion, opt)
    self.model = model
    self.criterion = criterion
    self.opt = opt
    self.optimState = opt.optimState
    
    self.input = nil
    self.target = nil

    self.params, self.gradParams = model:getParameters()
    self.feval = function() return self.err, self.gradParams end

    self.util = require 'utils'(opt)
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
        --self:copyInputs(sample, 'train')
        self.input = sample.input:cuda()
        self.target = sample.target:cuda()
        sample = nil
        collectgarbage()
        collectgarbage()

        self.model:zeroGradParameters()
        self.model:forward(self.input)
        self.criterion(self.model.output, self.target)
        if self.criterion.output >= self.opt.mulImg^2 then
            print('skipping samples with exploding error')
        elseif self.criterion.output ~= self.criterion.output then
            print('skipping samples with nan error')
        else
            err = err + self.criterion.output
            self.model:backward(self.input, self.criterion.gradInput)
            if self.opt.clip > 0 then
                self.gradParams:clamp(-self.opt.clip / self.opt.lr, self.opt.clip / self.opt.lr)
            end
            self.optimState.method(self.feval, self.params, self.optimState)
            iter = iter + 1
        end
        
        trainTime = trainTime + trainTimer:time().real
        if n % self.opt.printEvery == 0 then
            local it = (epoch - 1) * self.opt.testEvery + n
            local lr_f, lr_d = self:get_lr()
            print(('[Iter: %.1fk][lr: %.2fe%d]\tTime: %.2f (data: %.2f)\terr: %.6f')
                :format(it / 1000, lr_f, lr_d, trainTime, dataTime, err / iter))
            if n % self.opt.testEvery ~= 0 then
                err, iter = 0, 0
            end
            trainTime, dataTime = 0, 0
        end

        if n % self.opt.testEvery == 0 then
            break
        end

        trainTimer:reset()
        dataTimer:reset()
    end
    if epoch % self.opt.manualDecay == 0 then
        local prevlr = self.optimState.learningRate
        self.optimState.learningRate = prevlr / 2
        print(string.format('Learning rate decreased: %.6f -> %.6f',
        prevlr, self.optimState.learningRate))
    end
    
    return err / iter
end

function Trainer:test(epoch, dataloader)
    local timer = torch.Timer()
    local iter, avgPSNR = 0, 0

    self.model:clearState()
    self.model:evaluate()
    collectgarbage()
    collectgarbage()

    cudnn.fastest = false
    cudnn.benchmark = false
    
    for n, sample in dataloader:run() do
        --self:copyInputs(sample,'test')
        self.input = sample.input:cuda()
        self.target = sample.target:cuda()
        sample = nil
        collectgarbage()
        collectgarbage()

        local input = nn.Unsqueeze(1):cuda():forward(self.input)
        if self.opt.nChannel == 1 then
            input = nn.Unsqueeze(1):cuda():forward(input)
        end
        local output = self.util:recursiveForward(input, self.model):squeeze(1)

        self.util:quantize(output, self.opt.mulImg)
        self.target:div(self.opt.mulImg)
        avgPSNR = avgPSNR + self.util:calcPSNR(output, self.target, self.opt.scale)
        
        image.save(paths.concat(self.opt.save, 'result', n .. '.png'), output) 

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

function Trainer:get_lr()
    local logLR = math.log(self.optimState.learningRate, 10)
    local characteristic = math.floor(logLR)
    local mantissa = logLR - characteristic
    local frac = math.pow(10,mantissa)

    return frac, characteristic
end


return M.Trainer
