local image = require 'image'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('sr.Trainer', M)

function Trainer:__init(model, criterion, opt)
    self.model = model
    self.tempModel = nil
    self.criterion = criterion

    self.opt = opt
    self.scale = opt.scale
    self.optimState = opt.optimState
    self.optim = nil
    if opt.optimMethod == 'SGD' then 
        self.optim = optim.sgd
    elseif opt.optimMethod == 'ADADELTA' then
        self.optim = optim.adadelta
    elseif opt.optimMethod == 'ADAM' then
        self.optim = optim.adam
    elseif opt.optimMethod == 'RMSPROP' then
        self.optim = optim.rmsprop
    else
        error('unknown optimization method')
    end  

    self.iter = opt.lastIter        --Total iterations

    self.input = nil
    self.target = nil
    self.params = nil
    self.gradParams = nil

    self.feval = function() return self.errB, self.gradParams end
    self.util = require 'utils'(opt)

    self.retLoss, self.retPSNR = nil, nil
    self.maxPerf, self.maxIdx = {}, {}
    for i = 1, #self.scale do
        table.insert(self.maxPerf, -1)
        table.insert(self.maxIdx, -1)
    end
end

function Trainer:train(epoch, dataloader)
    local size = dataloader:size()
    local trainTimer = torch.Timer()
    local dataTimer = torch.Timer()
    local trainTime, dataTime = 0, 0
    local globalIter, globalErr, localErr = 0, 0, 0

    local pe = self.opt.printEvery
    local te = self.opt.testEvery

    cudnn.fastest = true
    cudnn.benchmark = true

    self.model:clearState()
    self.model:cuda()
    if self.opt.nGPU == 1 then
        self:prepareSwap('cuda')
    end
    self.model:training()
    self:getParams()
    collectgarbage()
    collectgarbage()

    for n, batch in dataloader:run() do
        dataTime = dataTime + dataTimer:time().real
        self:copyInputs(batch.input, batch.target, 'train')
        local scaleIdx = batch.scaleIdx

        self.model:zeroGradParameters()
        if self.opt.nGPU == 1 then
            --Fast model swap
            self.tempModel = self.model
            self.model = self.swapTable[scaleIdx]
        end

        self.model:forward(self.input.train)
        self.criterion(self.model.output, self.target)
        self.model:backward(self.input.train, self.criterion.gradInput)

        if self.opt.nGPU == 1 then
            --Return to original model
            self.model = self.tempModel
        end
        
        self.iter = self.iter + 1
        globalIter = globalIter + 1
        globalErr = globalErr + self.criterion.output
        localErr = localErr + self.criterion.output

        if self.opt.clip > 0 then
            self.gradParams:clamp(-self.opt.clip / self.opt.lr, self.opt.clip / self.opt.lr)
        end

        self:calcLR()
        self.optim(self.feval, self.params, self.optimState)
        trainTime = trainTime + trainTimer:time().real
        
        if n % pe == 0 then
            local lr_f, lr_d = self:lrPrint()
            print(('[Iter: %.1fk / lr: %.2fe%d] \tTime: %.2f (Data: %.2f) \tErr: %.6f')
                :format(self.iter / 1000, lr_f, lr_d, trainTime, dataTime, localErr / pe))
            localErr, trainTime, dataTime = 0, 0, 0
        end

        trainTimer:reset()
        dataTimer:reset()

        if n % te == 0 then
            break
        end
    end

    self.retLoss = globalErr / globalIter
end

function Trainer:test(epoch, dataloader)
    --Code for multiscale learning
    local timer = torch.Timer()
    local iter, avgPSNR = 0, {}
    for i = 1, #self.scale do
        table.insert(avgPSNR, 0)
    end

    cudnn.fastest = false
    cudnn.benchmark = false

    self.model:clearState()
    if self.opt.nGPU == 1 then
        self.modelTest = self.model
    else
        self.modelTest = self.modelTest or self.model:get(1)
    end
    self.modelTest:evaluate()
    if self.opt.nGPU == 1 then
        self:prepareSwap('cuda')
    end
    collectgarbage()
    collectgarbage()
    
    for n, batch in dataloader:run() do
        for i = 1, #self.scale do
            local sc = self.scale[i]
            self:copyInputs(batch.input[i], batch.target[i], 'test')

            local input = nn.Unsqueeze(1):cuda():forward(self.input.test)
            if self.opt.nChannel == 1 then
                input = nn.Unsqueeze(1):cuda():forward(input)
            end
            
            if self.opt.nGPU == 1 then    
                --Fast model swap
                self.tempModel = self.modelTest
                self.modelTest = self.swapTable[i]
            end

            local output = self.util:chopForward(input, self.modelTest, self.scale[i], self.opt.chopShave, self.opt.chopSize)

            if self.opt.nGPU == 1 then
                --Return to original model
                self.modelTest = self.tempModel
            end

            if self.opt.selOut > 0 then
                output = output[selOut]
            end

            output = output:squeeze(1)
            self.util:quantize(output, self.opt.mulImg)
            self.target:div(self.opt.mulImg)
            avgPSNR[i] = avgPSNR[i] + self.util:calcPSNR(output, self.target, sc)

            image.save(paths.concat(self.opt.save, 'result', n .. '_X' .. sc .. '.png'), output) 

            iter = iter + 1
            
            self.modelTest:clearState()
            self.input.test = nil
            self.target = nil
            output = nil
            collectgarbage()
            collectgarbage()
        end
        batch = nil
        collectgarbage()
        collectgarbage()
    end
    print(('[Epoch %d (iter/epoch: %d)] Test time: %.2f')
        :format(epoch, self.opt.testEvery, timer:time().real))

    for i = 1, #self.scale do
        avgPSNR[i] = avgPSNR[i] * #self.scale / iter
        if avgPSNR[i] > self.maxPerf[i] then
            self.maxPerf[i] = avgPSNR[i]
            self.maxIdx[i] = epoch
        end
        print(('(scale %d) Average PSNR: %.4f (Highest ever: %.4f at epoch = %d)')
            :format(self.scale[i], avgPSNR[i], self.maxPerf[i], self.maxIdx[i]))
    end
    print('')
    
    self.retPSNR = avgPSNR
end

function Trainer:copyInputs(input, target, mode)
    self.input = {}
    if mode == 'train' then
        self.input.train = self.input.train or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
        self.input.train:resize(input:size()):copy(input)
    elseif mode == 'test' then
        self.input.test = self.input.test or torch.CudaTensor()
        self.input.test:resize(input:size()):copy(input)
    end

    self.target = self.target or torch.CudaTensor()
    self.target:resize(target:size()):copy(target)

    input = nil
    target = nil
    collectgarbage()
    collectgarbage()
end

function Trainer:lrPrint()
    local logLR = math.log(self.optimState.learningRate, 10)
    local characteristic = math.floor(logLR)
    local mantissa = logLR - characteristic
    local frac = math.pow(10,mantissa)

    return frac, characteristic
end

function Trainer:calcLR()
    local iter, halfLife = self.iter, self.opt.halfLife
    local lr
    if self.opt.lrDecay == 'step' then -- decay lr by half periodically
        local nStep = math.floor((iter - 1) / halfLife)
        lr = self.opt.lr / math.pow(2, nStep)
    elseif self.opt.lrDecay == 'exp' then -- decay lr exponentially. y = y0 * e^(-kt)
        local k = math.log(2) / halfLife
        lr = self.opt.lr * math.exp(-k * iter)
    elseif self.opt.lrDecay == 'inv' then -- decay lr as y = y0 / (1 + kt)
        local k = 1 / halfLife
        lr = self.opt.lr / (1 + k * iter)
    end

    self.optimState.learningRate = lr
end

function Trainer:getParams()
    self.params, self.gradParams = self.model:getParameters()
end

function Trainer:prepareSwap(modelType)
    self.swapTable = {}
    for i = 1, #self.scale do
        local swapped = self.util:swapModel(self.model, i)
        if modelType == 'float' then
            swapped = swapped:float()
        elseif modelType == 'cuda' then
            swapped = swapped:cuda()
        end
        table.insert(self.swapTable, swapped)
    end
end

function Trainer:updateLoss(loss)
    table.insert(loss, {key = self.iter, value = self.retLoss})

    return loss
end

function Trainer:updatePSNR(psnr)
    for i = 1, #self.scale do
        table.insert(psnr[i], {key = self.iter, value = self.retPSNR[i]})
    end

    return psnr
end

function Trainer:updateLR(lr)
    table.insert(lr, {key = self.iter, value = self.optimState.learningRate})

    return lr
end

return M.Trainer
