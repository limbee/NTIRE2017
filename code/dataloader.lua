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
    self.area = dataset.area
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
    local patchSize,scale = self.opt.patchSize,self.opt.scale
    local nChannel = self.opt.nChannel

    local idx, sample = 1, nil
    local function enqueue()
        if self.split == 'train' then
            while threads:acceptsjob() do
                local indices
                if batchSize > size-idx+1 then
                    idx = 1
                    perm = torch.randperm(size)
                end
                indices = perm:narrow(1, idx, batchSize)
                threads:addjob(
                    function(indices)
                        --local batchSize = indices:size(1)
                        local tarSize = patchSize
                        local inpSize = netType=='VDSR' and patchSize or patchSize/scale
                        local input_batch = torch.Tensor(batchSize,nChannel,inpSize,inpSize):zero()
                        local target_batch = torch.Tensor(batchSize,nChannel,tarSize,tarSize):zero()

                        for i,index in ipairs(indices:totable()) do
                            local idx_ = index
                            ::redo::
                            local sample = _G.dataset:get(idx_)
                            if not sample then 
                                idx_ = torch.random(size)
                                goto redo
                            end

                            sample = _G.augment(sample)

                            input_batch[i]:copy(sample.input)
                            target_batch[i]:copy(sample.target)
                        end
                        collectgarbage()
                        collectgarbage()
                        return {
                            input = input_batch,
                            target = target_batch,
                        }
                    end,
                    function (_sample_)
                        sample = _sample_
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
                        return {
                            input = sample.input,
                            target = sample.target
                        }
                    end,
                    function (_sample_)
                        sample = _sample_
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
