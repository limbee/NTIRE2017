local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('sr.DataLoader', M)

function DataLoader.create(opt)
    print('loading data...')
    local loaders = {}
    for i, split in ipairs{'train', 'val'} do
        print('\tInitializing data loader for ' .. split .. ' set...')
        local dataset = require('data/' .. opt.dataset)(opt, split)
        loaders[i] = M.DataLoader(dataset, opt, split)
    end
    return unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
    self.opt = opt
    self.split = split

    local manualSeed = self.opt.manualSeed
    local function init()
        require('data/' .. opt.dataset)
        torch.setdefaulttensortype('torch.FloatTensor')
    end
    local function main(idx)
        if manualSeed ~= 0 then
            torch.manualSeed(manualSeed + idx)
        end
        _G.dataset = dataset
        _G.augment = dataset:augment()
        _G.scale = opt.scale
        return dataset:__size()
    end

    local threads, sizes = Threads(self.opt.nThreads, init, main)
    self.threads = threads
    self.__size = sizes[1][1]

    self.batchSize = self.opt.batchSize
    self.nChannel = self.opt.nChannel
    self.patchSize = self.opt.patchSize
    self.multiPatch = self.opt.multiPatch
    self.dataSize = self.opt.dataSize
    self.scale = self.opt.scale
end

function DataLoader:size()
    return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
    local threads = self.threads
    threads:synchronize()

    local size = self.__size
    local batchSize, nChannel, patchSize = self.batchSize, self.nChannel, self.patchSize
    local multiPatch = self.multiPatch
    local dataSize = self.dataSize
    local perm = torch.randperm(size)

    local idx, batch = 1, nil

    local function enqueue()
        if self.split == 'train' then
            while threads:acceptsjob() do
                --Shuffle the indices
                if batchSize > (size - idx + 1) then
                    idx = 1
                    perm = torch.randperm(size)
                end
                local indices = perm:narrow(1, idx, batchSize)

                threads:addjob(
                    function(indices)
                        local scaleIdx = torch.random(1, #_G.scale)
                        local scale = _G.scale[scaleIdx]
                        local tarSize = multiPatch and (patchSize * scale) or patchSize
                        local inpSize = (dataSize == 'big') and tarSize or tarSize / scale

                        local input = torch.Tensor(batchSize, nChannel, inpSize, inpSize)
                        local target = torch.Tensor(batchSize, nChannel, tarSize, tarSize)

                        for i = 1, batchSize do
                            local sample = nil
                            repeat
                                --Code for multiscale learning
                                sample = _G.dataset:get(indices[i], scaleIdx)
                                indices[i] = torch.random(size)
                            until sample

                            sample = _G.augment(sample)
                            input[i]:copy(sample.input)
                            target[i]:copy(sample.target)
                            sample = nil
                        end
                        collectgarbage()
                        collectgarbage()

                        return {
                            input = input,
                            target = target,
                            scaleIdx = scaleIdx 
                        }    
                    end,
                    function (_batch_)
                        batch = _batch_
                        _batch_ = nil
                        collectgarbage()
                        collectgarbage()

                        return batch
                    end,
                    indices
                )
                idx = idx + batchSize
            end
        elseif self.split == 'val' then
            while idx <= size and threads:acceptsjob() do
                threads:addjob(
                    function(idx)
                        local _inputVal, _targetVal = {}, {}
                        for i = 1, #_G.scale do
                            local sample = _G.dataset:get(idx, i)
                            table.insert(_inputVal, sample.input)
                            table.insert(_targetVal, sample.target)
                        end
                        collectgarbage()
                        collectgarbage()

                        return {
                            input = _inputVal,
                            target = _targetVal
                        }
                    end,
                    function (_batch_)
                        batch = _batch_
                        _batch_ = nil
                        collectgarbage()
                        collectgarbage()

                        return batch
                    end,
                    idx
                )        
                idx = idx + 1
            end
        end 
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
            return nil
        end
        threads:dojob()
        if threads:haserror() then
            threads:synchronize()
        end
        enqueue()
        n = n + 1

        return n, batch
    end

    return loop
end

return M.DataLoader
