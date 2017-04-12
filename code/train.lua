local image = require 'image'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('sr.Trainer', M)

function Trainer:__init(model, criterion, opt)
    self.model = model
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
    self.currentErr = nil

    self.feval = function() return self.currentErr, self.gradParams end
    self.util = require 'utils'(opt)

    self.retLoss, self.retPSNR = 1e9, 1e9
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

    local isSwap = self.opt.isSwap
    local swapTable = self:prepareSwap(isSwap)
    local tempModel = nil

    cudnn.fastest = true
    cudnn.benchmark = true

    self.model:clearState()


    self.model:training()
    self:getParams()
    collectgarbage()
    collectgarbage()

    for n, batch in dataloader:run() do
        dataTime = dataTime + dataTimer:time().real
        self:copyInputs(batch.input, batch.target, 'train')
        local scaleIdx = batch.scaleIdx

        self.model:zeroGradParameters()
        if isSwap then
            --Fast model swap
            tempModel = self.model
            --self.model = swapTable[scaleIdx]
            self.model = self.util:swapModel(self.model, scaleIdx)
        end

        self.model:forward(self.input.train)
        self.currentErr = self.criterion(self.model.output, self.target)
        self.model:backward(self.input.train, self.criterion.gradInput)

        if isSwap then
            --Return to original model
            self.model = tempModel
        end

        -- If the error is larger than skipBatch * (previous error),
        -- do not use it to update the parameters.
        if self.currentErr < self.retLoss * self.opt.skipBatch then
            self.iter = self.iter + 1
            globalIter = globalIter + 1
            globalErr = globalErr + self.currentErr
            localErr = localErr + self.currentErr

            if self.opt.clip > 0 then
                self.gradParams:clamp(-self.opt.clip / self.opt.lr, self.opt.clip / self.opt.lr)
            end

            self:calcLR()
            self.optim(self.feval, self.params, self.optimState)
        else
            print(('Warning: Error is too large! Skip this batch. (Err: %.6f)'):format(self.currentErr))
        end

        trainTime = trainTime + trainTimer:time().real
        
        if self.iter % pe == 0 then
            local lr_f, lr_d = self:lrPrint()
            print(('[Iter: %.1fk / lr: %.2fe%d] \tTime: %.2f (Data: %.2f) \tErr: %.6f')
                :format(self.iter / 1000, lr_f, lr_d, trainTime, dataTime, localErr / pe))
            localErr, trainTime, dataTime = 0, 0, 0
        end

        trainTimer:reset()
        dataTimer:reset()

        if self.iter % te == 0 then
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

    local isSwap = self.opt.isSwap
    local swapTable = self:prepareSwap(isSwap)
    local tempModel = nil

    cudnn.fastest = false
    cudnn.benchmark = false

    self.model:clearState()
    local modelTest = nil

    if self.opt.nGPU == 1 then
        modelTest = self.model
    else
        modelTest = self.model:get(1)
    end
    modelTest:evaluate()

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

            if isSwap then    
                --Fast model swap
                tempModel = modelTest
                --modelTest = swapTable[i]
                modelTest = self.util:swapModel(tempModel, i)
            end

            local output = self.util:chopForward(input, modelTest, self.scale[i], self.opt.chopShave, self.opt.chopSize)

            if isSwap then
                --Return to original model
                modelTest = tempModel
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
            
            modelTest:clearState()
            self.input.test = nil
            self.target = nil
            output = nil
            collectgarbage()
            collectgarbage()
        end
    end
    
    modelTest = nil
    collectgarbage()
    collectgarbage()

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

function Trainer:prepareSwap(isSwap)
    local ret = {}
    if isSwap then
        for i = 1, #self.scale do
            table.insert(ret, self.util:swapModel(self.model, i))
        end
    end
    return ret
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
