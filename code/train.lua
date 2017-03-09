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

    self.err = 0
    self.params, self.gradParams = model:getParameters()
    self.feval = function() return self.err, self.gradParams end
end

function Trainer:train(epoch, dataloader)
    cudnn.fastest = true
    cudnn.benchmark = true

    local size = dataloader:size()
    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local trainTime, dataTime = 0,0
    local iter, err = 0,0

    self.model:training()
    for n, sample in dataloader:run() do
        dataTime = dataTime + dataTimer:time().real
        
        self:copyInputs(sample,'train') -- Copy input and target to the GPU

        self.model:zeroGradParameters()
        self.model:forward(self.input)
        self.err = self.criterion(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)
        if self.opt.clip > 0 then
            self.gradParams:clamp(-self.opt.clip/self.opt.lr,self.opt.clip/self.opt.lr)
        end
        self.optimState.method(self.feval, self.params, self.optimState)

        err = err + self.err

        trainTime = trainTime + timer:time().real
        timer:reset()
        dataTimer:reset()

        iter = iter + 1
        if n % self.opt.printEvery == 0 then
            local it = (epoch-1)*self.opt.testEvery + n
            if it>1000 then it = string.format('%.1fk',it/1000) end
            print(('[Iter: ' .. it .. '] Time: %.3f (data: %.3f),\terr: %.6f')
                :format(trainTime, dataTime, err/iter))
            if n % self.opt.testEvery ~= 0 then
                err, iter = 0,0
                trainTime, dataTime = 0,0
            end
        end

        if (n % self.opt.testEvery == 0) then
            break
        end
    end
    
    return err/iter
end

function Trainer:test(epoch, dataloader)
    local timer = torch.Timer()
    local iter,avgPSNR = 0,0

    self.model:evaluate()
    -- Following cudnn settings are to prevent cudnn errors 
    -- occasionally occur during testing.
    cudnn.fastest = false
    cudnn.benchmark = false
    for n, sample in dataloader:run() do
        self:copyInputs(sample,'test')

        local input, target = self.input, self.target
        input = nn.Unsqueeze(1):cuda():forward(input)
        if self.opt.nChannel==1 then
            input = nn.Unsqueeze(1):cuda():forward(input)
        end

        self.model:clearState()
        collectgarbage()
        collectgarbage()
        local model = self.model:clone('weight','bias')

        if torch.type(model)=='nn.DataParallelTable' then model = model:get(1) end
        local __model = model

        -- This function prevents the gpu memory from overflowing
        -- by passing the input layer-by-layer through the network.
        local function recursiveForward(input, m)
            local output
            if (m.__typename:find('Concat')) then
                output = {}
                for i = 1, m:size() do
                    table.insert(output, recursiveForward(input, m:get(i)))
                end
            elseif (m.__typename:find('Sequential')) then
                output = input
                for i = 1, #m do
                    output = recursiveForward(output, m:get(i))
                end
            elseif (m.__typename:find('Identity')) then
                if (type(input) == 'table') then

                else
                    output = m:forward(input):clone()
                end
                m = nil
                __model:clearState()
                collectgarbage()
                collectgarbage()
            else
                output = m:forward(input):clone()
                m = nil
                __model:clearState()
                collectgarbage()
                collectgarbage()
            end
            input = nil
            collectgarbage()
            collectgarbage()
            return output
        end
        local output = recursiveForward(input,model):squeeze(1)
        if (self.opt.netType == 'bandnet') then
            output = output[2]
        end
        avgPSNR = avgPSNR + util:calcPSNR(output, target, self.opt.scale)
        image.save(paths.concat(self.opt.save, 'result', n .. '.png'), output:float():squeeze():div(255))

        iter = iter + 1
        collectgarbage()
        collectgarbage()
    end

    print(('[epoch %d (iter/epoch: %d)] Average PSNR: %.2f,  Test time: %.2f\n')
        :format(epoch, self.opt.testEvery, avgPSNR / iter, timer:time().real))

    return avgPSNR / iter
end

function Trainer:copyInputs(sample, mode)
    if (mode == 'train') then
        self.input = self.input or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
    elseif (mode == 'test') then
        self.input = self.input or torch.CudaTensor()
    end

    self.input:resize(sample.input:size()):copy(sample.input)
    self.target = self.target or torch.CudaTensor()
    self.target:resize(sample.target:size()):copy(sample.target)
end

return M.Trainer
