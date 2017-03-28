local image = require 'image'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('sr.Trainer', M)

function Trainer:__init(model, criterion, opt)
    self.model = model
    self.criterion = criterion
    self.opt = opt
    self.optimState = opt.optimState
    self.iter = opt.lastIter
    
    self.input = nil
    self.target = nil

    self.params = nil
    self.gradParams = nil

    self.feval = function() return self.err, self.gradParams end

    self.util = require 'utils'(opt)

    self.errThreshold = 
        (
            (opt.abs + opt.chbn + opt.smoothL1) * opt.mulImg +
            opt.mse * opt.mulImg^2 +
            opt.ssim + 
            opt.band * opt.mulImg +
            (opt.grad + opt.grad2*2) * (opt.gradDist == 'mse' and opt.mulImg^2 or opt.mulImg) +
            opt.gradPrior * opt.mulImg^opt.gradPower +
            opt.fd * opt.mulImg
        ) * 2   -- x2 margin
    self.lastErr = math.huge
end

function Trainer:train(epoch, dataloader)
    local size = dataloader:size()
    local trainTimer = torch.Timer()
    local dataTimer = torch.Timer()
    local trainTime, dataTime = 0, 0
    local iter, err = 0, 0
    local globalIter, globalErr = 0, 0
    
    cudnn.fastest = true
    cudnn.benchmark = true

    self.model:clearState()
    self.model:cuda()
    self.model:training()
    self:getParams()
    collectgarbage()
    collectgarbage()

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

        if self.criterion.output >= self.errThreshold or 
            self.criterion.output >= self.lastErr *2 then
            print('skipping this minibatch with exploding error')
        elseif self.criterion.output ~= self.criterion.output then
            print('skipping this minibatch with nan error')
        else
            self.model:backward(self.input, self.criterion.gradInput)
            if self.opt.clip > 0 then
                self.gradParams:clamp(-self.opt.clip / self.opt.lr, self.opt.clip / self.opt.lr)
            end
            self.optimState.method(self.feval, self.params, self.optimState)

            err = err + self.criterion.output
            iter = iter + 1
            globalErr = globalErr + self.criterion.output
            globalIter = globalIter + 1
        end
        self.iter = self.iter + 1
        
        trainTime = trainTime + trainTimer:time().real
        if n % self.opt.printEvery == 0 then
            local lr_f, lr_d = self:get_lr()
            print(('[Iter: %.1fk][lr: %.2fe%d]\tTime: %.2f (data: %.2f)\terr: %.6f')
                :format(self.iter / 1000, lr_f, lr_d, trainTime, dataTime, err / iter))
            err, iter = 0, 0
            trainTime, dataTime = 0, 0
        end

        trainTimer:reset()
        dataTimer:reset()

        if n % self.opt.testEvery == 0 then
            break
        end
    end

    if epoch % self.opt.manualDecay == 0 then
        local prevlr = self.optimState.learningRate
        self.optimState.learningRate = prevlr / 2
    end
    
    self.lastErr = globalErr / globalIter
    return self.lastErr
end

function Trainer:test(epoch, dataloader)
    local timer = torch.Timer()
    local iter, avgPSNR = 0, 0

    cudnn.fastest = false
    cudnn.benchmark = false

    self.model:clearState()
    self.model:float()
    self.model:evaluate()
    collectgarbage()
    collectgarbage()
    
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
        local output = self.util:recursiveForward(input, self.model, self.opt.safe):squeeze(1)

        self.util:quantize(output, self.opt.mulImg)
        self.target:div(self.opt.mulImg)
        avgPSNR = avgPSNR + self.util:calcPSNR(output, self.target, self.opt.scale)

        image.save(paths.concat(self.opt.save, 'result', n .. '.png'), output) 

        iter = iter + 1
        self.model:clearState()
        output = nil
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

function Trainer:get_iter()
    return self.iter
end

function Trainer:getParams()
    self.params, self.gradParams = self.model:getParameters()
end


return M.Trainer
