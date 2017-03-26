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
    
    self.iter, self.err = 0, 0
    local dsc = dataloader.scale

    cudnn.fastest = true
    cudnn.benchmark = true

    self.model:training()
    for n, sample in dataloader:run() do
        dataTime = dataTime + dataTimer:time().real
        self:copyInputs(sample, 'train')

        self.model:zeroGradParameters()
        self.model:forward(self.input)

        --Code for multiscale learning
        local sci = 1
        if #dsc > 1 then
            local bs, _, th, tw = table.unpack(self.target:size():totable())
            for i = 1, #dsc do
                local bs, _, oh, ow = table.unpack(self.model.output[i]:size():totable())
                if (oh == th) and (ow == tw) then
                    sci = i
                    break;
                end
            end
            self.criterion(self.model.output[sci], self.target)
        else
            self.criterion(self.model.output, self.target)
        end

        if self.criterion.output >= self.opt.mulImg^2 then
            print('skipping samples with exploding error')
        elseif self.criterion.output ~= self.criterion.output then
            print('skipping samples with nan error')
        else
            self.err = self.err + self.criterion.output
            
            --Code for multiscale learning
            if #dsc > 1 then
                local gi = {}
                for i = 1, #dsc do
                    if i == sci then
                        table.insert(gi, self.criterion.gradInput)
                    else
                        table.insert(gi, torch.zeros(self.output[i]:size()):cuda())
                    end
                end
                self.model:backward(self.input, gi)
            else
                self.model:backward(self.input, self.criterion.gradInput)
            end

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
            local lr_f, lr_d = self:get_lr()
            print(('[Iter: %.1fk][lr: %.2fe%d]\tTime: %.2f (data: %.2f)\terr: %.6f')
                :format(it / 1000, lr_f, lr_d, trainTime, dataTime, self.err / self.iter))
            if n % self.opt.testEvery ~= 0 then
                self.err, self.iter = 0, 0
            end
            trainTime, dataTime = 0, 0

            --This code periodically cleans up GPU memory when we use reTrain table
            if self.opt.reTrain > 0 then
                local capacity = self.opt.batchSize * self.opt.reTrain
                local sz = #self.reTable
                if sz > capacity then
                    table.sort(self.reTable, function(a, b) return a.err > b.err end)
                    for i = (capacity + 1), sz do
                        table.remove(self.reTable)
                    end
                    collectgarbage()
                    collectgarbage()
                end
            end
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
    --Code for multiscale learning
    local timer = torch.Timer()
    local iter, avgPSNR = 0, {}
    local dsc = dataloader.scale

    for i = 1, #dsc do
        table.insert(avgPSNR, 0)
    end

    self.model:clearState()
    self.model:evaluate()
    collectgarbage()
    collectgarbage()

    cudnn.fastest = false
    cudnn.benchmark = false
        
    for n, sample in dataloader:run() do
        for i = 1, #dsc do
            local sc = dsc[i]

            self.input = self.input or torch.CudaTensor()
            self.target = self.target or torch.CudaTensor()
            self.input:resize(sample.input[i]:size()):copy(sample.input[i])
            self.target:resize(sample.target[i]:size()):copy(sample.target[i])
            sample.input[i] = nil
            sample.target[i] = nil
                                                          
            local input = nn.Unsqueeze(1):cuda():forward(self.input)
            if self.opt.nChannel == 1 then
                input = nn.Unsqueeze(1):cuda():forward(input)
            end
            local output = self.util:recursiveForward(input, self.model)
            
            if self.opt.nOut > 1 then
                local selOut = (self.opt.selOut > 0) and self.opt.selOut or i
                output = output[selOut]
            end

            output = output:squeeze(1)
            self.util:quantize(output, self.opt.mulImg)
            self.target:div(self.opt.mulImg)
            avgPSNR[i] = avgPSNR[i] + self.util:calcPSNR(output, self.target, sc)

            image.save(paths.concat(self.opt.save, 'result', n .. '_X' .. sc .. '.png'), output) 

            iter = iter + 1
            
            self.model:clearState()
            self.input = nil
            self.target = nil
            output = nil
            collectgarbage()
            collectgarbage()
        end
        sample = nil
        collectgarbage()
        collectgarbage()
    end
    print(('epoch %d (iter/epoch: %d)] Test time: %.2f')
        :format(epoch, self.opt.testEvery, timer:time().real))

    for i = 1, #dsc do
        avgPSNR[i] = avgPSNR[i] / iter
        print(('Average PSNR: %.4f (X%d)')
            :format(avgPSNR[i], dsc[i]))
    end
        
    return avgPSNR
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
                --image.save('tools/retrained/' .. idx .. '_in.png', inputBatch[j] / self.opt.mulImg)
                --image.save('tools/retrained/' .. idx .. '_GT.png', targetBatch[j] / self.opt.mulImg)
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

function Trainer:get_lr()
    local logLR = math.log(self.optimState.learningRate, 10)
    local characteristic = math.floor(logLR)
    local mantissa = logLR - characteristic
    local frac = math.pow(10,mantissa)

    return frac, characteristic
end


return M.Trainer
