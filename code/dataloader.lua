local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('sr.DataLoader', M)

function DataLoader.create(opt)
    print('loading data...')
    local loaders = {}
    for i, split in ipairs{'train', 'val'} do
        local dataset = require('data/' .. opt.dataset)(opt,split)
        print('\tInitializing data loader for ' .. split .. ' set...')
        loaders[i] = M.DataLoader(dataset, opt, split)
    end
    return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
    local manualSeed = opt.manualSeed
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
        return dataset:__size()
    end

    local threads, sizes = Threads(opt.nThreads,init,main)
    self.threads = threads
    self.__size = sizes[1][1]
    self.batchSize = opt.batchSize
    self.split = split
    self.opt = opt
end

function DataLoader:size()
    return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
    local threads = self.threads
    threads:synchronize()
    local size, batchSize = self.__size, self.batchSize
    local perm = torch.randperm(size)
    local netType = self.opt.netType
    local dataSize = self.opt.dataSize
    local patchSize, scale = self.opt.patchSize, self.opt.scale
    local tarSize = patchSize
    local inpSize = (dataSize == 'big') and patchSize or patchSize / scale
    local nChannel = self.opt.nChannel

    local idx, sample = 1, nil

    local function enqueue()
        if self.split == 'train' then
            while threads:acceptsjob() do
                if batchSize > (size - idx + 1) then
                    idx = 1
                    perm = torch.randperm(size)
                end
                local indices = perm:narrow(1, idx, batchSize)
                threads:addjob(
                    function(indices)
                        local inputBatch = torch.zeros(batchSize, nChannel, inpSize, inpSize)
                        local targetBatch = torch.zeros(batchSize, nChannel, tarSize, tarSize)

                        for i = 1, batchSize do
                            local sample = nil
                            local si = i
                            repeat
                                sample = _G.dataset:get(indices[si])
                                si = torch.random(size)
                            until sample

                            sample = _G.augment(sample)
                            inputBatch[i]:copy(sample.input)
                            targetBatch[i]:copy(sample.target)
                            sample = nil
                        end
                        collectgarbage()
                        collectgarbage()

                        return {
                            input = inputBatch,
                            target = targetBatch,
                        }    
                    end,
                    function (_sample_)
                        sample = _sample_
                        _sample_ = nil
                        collectgarbage()
                        collectgarbage()

                        return sample
                    end,
                    indices
                )
                idx = idx + batchSize
            end
        elseif self.split == 'val' then
            while idx <= size and threads:acceptsjob() do
                threads:addjob(
                    function(idx)
                        local sample = _G.dataset:get(idx)
                        local ret = {
                            input = sample.input:clone(),
                            target = sample.target:clone()
                        }
                        sample = nil
                        collectgarbage()
                        collectgarbage()

                        return ret
                    end,
                    function (_sample_)
                        sample = _sample_
                        _sample_ = nil
                        collectgarbage()
                        collectgarbage()

                        return sample
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

        return n, sample
    end

    return loop
end

return M.DataLoader
