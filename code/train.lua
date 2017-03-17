local image = require 'image'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('sr.Trainer', M)

function Trainer:__init(model, criterion, opt)
    self.model = model
    self.criterion = criterion
    self.opt = opt
    self.optimState = opt.optimState
    
    self.iter = 0
    self.err = 0

    self.input = nil
    self.target = nil
    self.reTable = {}

    self.params, self.gradParams = model:getParameters()
    self.feval = function() return self.err, self.gradParams end

    self.util = require 'utils'(opt)
end

function Trainer:train(epoch, dataloader)
    local size = dataloader:size()
    local trainTimer = torch.Timer()
    local dataTimer = torch.Timer()
    local trainTime, dataTime = 0, 0
    
    self.iter = 0
    self.err = 0

    cudnn.fastest = true
    cudnn.benchmark = true

    self.model:training()
    for n, sample in dataloader:run() do
        dataTime = dataTime + dataTimer:time().real
        --Copy input and target to the GPU
        self:copyInputs(sample, 'train')

        self.model:zeroGradParameters()
        self.model:forward(self.input)
        self.criterion(self.model.output, self.target)
        if self.criterion.output >= self.opt.mulImg^2 then
            print('skipping samples with exploding error')
        else
            self.err = self.err + self.criterion.output
            self.model:backward(self.input, self.criterion.gradInput)
            if self.opt.clip > 0 then
                self.gradParams:clamp(-self.opt.clip / self.opt.lr, self.opt.clip / self.opt.lr)
            end
            self.optimState.method(self.feval, self.params, self.optimState)
            self.iter = self.iter + 1
        end
        
        if self.opt.reTrain > 0 then
            local bs = self.opt.batchSize
            local errors = (self.model.output - self.target):pow(2):view(bs, -1):mean(2):squeeze(2)
            for i = 1, bs do
                table.insert(self.reTable, {err = errors[i], input = self.input[i], target = self.target[i]})
            end
        end

        trainTime = trainTime + trainTimer:time().real
        
        if n % self.opt.printEvery == 0 then
            local it = (epoch - 1) * self.opt.testEvery + n
            print(('[Iter: %.1fk]\tTime: %.2f (data: %.2f)\terr: %.6f')
                :format(it / 1000, trainTime, dataTime, self.err / self.iter))
            if n % self.opt.testEvery ~= 0 then
                self.err, self.iter = 0, 0
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
        print(('Learning rate decreased: %.6f -> %.6f')
            :format(prevlr, self.optimState.learningRate))
    end
    
    return self.err / self.iter
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
        self:copyInputs(sample,'test')

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
        self.input = nil
        self.target = nil
        output = nil
        collectgarbage()
        collectgarbage()
    end

    print(('[epoch %d (iter/epoch: %d)] Average PSNR: %.4f,  Test time: %.2f\n')
        :format(epoch, self.opt.testEvery, avgPSNR / iter, timer:time().real))

    return avgPSNR / iter
end

function Trainer:reTrain()
    local rt = self.opt.reTrain
    if rt > 0 then
        local trainTimer = torch.Timer()
        trainTimer:reset()

        self.iter = 0
        self.err = 0

        table.sort(self.reTable, function(a, b) return a.err > b.err end)

        local idx = 1
        local bs = self.opt.batchSize
        local isz = self.reTable[1].input:size()
        local tsz = self.reTable[1].target:size()

        local inputBatch = torch.Tensor(bs, isz[1], isz[2], isz[3])
        local targetBatch = torch.Tensor(bs, tsz[1], tsz[2], tsz[3])
        for i = 1, self.opt.reTrain do
            for j = 1, bs do
                inputBatch[j]:copy(self.reTable[idx].input)
                targetBatch[j]:copy(self.reTable[idx].target)
                image.save('tools/retrained/' .. idx .. '_in.png', inputBatch[j] / self.opt.mulImg)
                image.save('tools/retrained/' .. idx .. '_GT.png', targetBatch[j] / self.opt.mulImg)
                idx = idx + 1
            end
            self:copyInputs({input = inputBatch, target = targetBatch}, 'train')
            self.model:zeroGradParameters()
            self.model:forward(self.input)
            self.criterion(self.model.output, self.target)
            self.err = self.err + self.criterion.output
            self.model:backward(self.input, self.criterion.gradInput)
            if self.opt.clip > 0 then
                self.gradParams:clamp(-self.opt.clip / self.opt.lr, self.opt.clip / self.opt.lr)
            end
            self.optimState.method(self.feval, self.params, self.optimState)
            self.iter = self.iter + 1
        end
        print(('Retrained %d batches\tTime: %.2f\terr: %.6f')
            :format(self.opt.reTrain, trainTimer:time().real, self.err / self.iter))
            
        for i = 1, #self.reTable do
            self.reTable[i].input = nil
            self.reTable[i].target = nil
            self.reTable[i] = nil
        end
        self.reTable = nil
        collectgarbage()
        collectgarbage()

        self.reTable = {}
    end
end

function Trainer:copyInputs(sample, mode)
    if mode == 'train' then
        self.input = self.input or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
    elseif mode == 'test' then
        self.input = self.input or torch.CudaTensor()
    end
    self.target = self.target or torch.CudaTensor()

    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)

    sample.input = nil
    sample.target = nil
    collectgarbage()
    collectgarbage()
end

return M.Trainer
