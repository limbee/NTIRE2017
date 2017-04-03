require 'nn'
require 'cunn'
require 'cudnn'
require 'image'

local M = {}
local util = torch.class('sr.util',M)

function util:__init(opt)
    if opt then
        self.opt = opt
        self.save = opt.save
    end
end

function util:plot(tbl, name, label)
    -- Assume tbl is a table of numbers, or a table of tables
    local fig = gnuplot.pdffigure(paths.concat(self.save, name .. '.pdf'))
    local label = label or name

    local function findMinMax(tb)
        local minKey, maxKey = math.huge, -math.huge    
        local minKeyValue, maxKeyValue
        for i = 1, #tb do
            if tb[i].key < minKey then
                minKey = tb[i].key
                minKeyValue = tb[i].value
            end
            if tb[i].key > maxKey then
                maxKey = tb[i].key
                maxKeyValue = tb[i].value
            end
        end
        return minKeyValue, maxKeyValue
    end

    local function typeTable(tb)
        for k, v in pairs(tb[1]) do
            if type(v) == 'table' then
                return 'table'
            else
                return 'number'
            end
        end
    end

    local function toTensor(tb)
        local xAxis = {}
        local yAxis = {}
        for i = 1, #tb do
            table.insert(xAxis, tb[i].key)
            table.insert(yAxis, tb[i].value)
        end
        return torch.Tensor(xAxis), torch.Tensor(yAxis)
    end

    local function __xAxisScale(xAxis)
        local maxIter = xAxis:max()
        if maxIter > 1e6 then
            return 1e6
        elseif maxIter > 1e3 then
            return 1e3
        else
            return 1
        end
    end

    local lines = {}
    local first, last
    local xAxisScale = 1
    if typeTable(tbl) ~= 'table' then -- single graph
        local xAxis, yAxis = toTensor(tbl)
        xAxisScale = __xAxisScale(xAxis)
        if name == 'Learning Rate' then
            yAxis:log():div(math.log(10))
        end
        table.insert(lines, {label, xAxis:div(xAxisScale), yAxis, '-'})
        first, last = findMinMax(tbl)
    else -- multiple lines
        assert(type(label) == 'table', 'label must be a table, if you want to draw lines more than 1')
        local tmp, _ = toTensor(tbl[1])
        xAxisScale = __xAxisScale(tmp)
        for i = 1, #tbl do
            local xAxis, yAxis = toTensor(tbl[i])
            table.insert(lines, {label[i], xAxis:div(xAxisScale), yAxis, '-'})
        end
        first, last = findMinMax(tbl[1])
    end

    gnuplot.plot(lines)
    if first < last then
        gnuplot.movelegend('right', 'bottom')
    else
        gnuplot.movelegend('right', 'top')
    end
    gnuplot.grid(true)
    gnuplot.title(name)
    local xlabel = 'Iterations'
    if xAxisScale > 1 then
        xlabel = xlabel .. ' (*1e' .. math.log(xAxisScale, 10) .. ')'
    end
    gnuplot.xlabel(xlabel)
    if name == 'Learning Rate' then
        -- gnuplot.raw('set logscale y') -- it doesn't seeem supported in torch
        -- gnuplot.raw('set format y "%.1t * 10^{%L}"')
        gnuplot.raw('set format y "%.1t"')
        gnuplot.ylabel('log10(LR)')
    end
	gnuplot.plotflush(fig)
	gnuplot.closeall()  
end

function util:checkpoint(model, criterion, loss, psnr, lr)
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    torch.save(paths.concat(self.save, 'model', 'model_' .. #loss .. '.t7'), model:clearState())
    torch.save(paths.concat(self.save, 'loss.t7'), loss)
    torch.save(paths.concat(self.save, 'psnr.t7'), psnr)
    torch.save(paths.concat(self.save, 'learning_rate.t7'), lr)
    torch.save(paths.concat(self.save, 'opt.t7'), self.opt)
end

function util:load()
    local ok, loss, psnr, lr
    local numLoss = 0

    if self.opt.load then
        ok, loss, psnr, lr, opt = pcall(
            function()
                local loss = torch.load(paths.concat(self.save, 'loss.t7'))
                local psnr = torch.load(paths.concat(self.save, 'psnr.t7'))
                local lr = torch.load(paths.concat(self.save, 'learning_rate.t7'))
                local opt = torch.load(paths.concat(self.save, 'opt.t7'))
                return loss, psnr, lr, opt 
            end)
        if ok then
            numLoss = #loss
            local _lastIter = loss[numLoss].key
            print(('Loaded history (%d epochs = %d iterations)\n'):format(numLoss, _lastIter))
            if self.opt.startEpoch > numLoss + 1 then
                error(('Start epoch cannot be bigger than history (%d epochs)'):format(numLoss))
            elseif self.opt.startEpoch == 1 then
                error('Please set -startEpoch bigger than 1, if you want to resume the training')
            elseif self.opt.startEpoch > 1 and self.opt.startEpoch <= numLoss then
                print(('Resuming the training from %d epoch'):format(self.opt.startEpoch))
                local _loss, _psnr, _lr = {}, {}, {}
                local lastIter = loss[#loss].key
                for i = 1, #loss do
                    if loss[i].key < lastIter then
                        table.insert(_loss, loss[i])
                        table.insert(_lr, lr[i])
                    end
                end
                for i = 1, #self.opt.scale do
                    table.insert(_psnr, {})
                    for j = 1, #psnr do
                        if psnr[i][j].key < lastIter then
                            table.insert(_psnr, psnr[i][j])
                        end
                    end
                end

                loss, psnr, lr = _loss, _psnr, _lr
                self.opt.lastIter = lastIter
            else -- This is the default setting. startEpoch = 0 corresponds to #loss + 1
                print(('Continue training (After %d epochs = %d iterations)'):format(numLoss, _lastIter))
                self.opt.startEpoch = numLoss + 1
                self.opt.lastIter = _lastIter
            end
        else
            error('history (loss, psnr, lr, options) does not exist')
        end
    else
        ok = false
        loss, psnr, lr = {}, {}, {}
        for i = 1, #self.opt.scale do
            table.insert(psnr, {})
        end
        self.opt.startEpoch = 1
        self.opt.lastIter = 0
    end

    return ok, loss, psnr, lr
end

function util:calcPSNR(output, target, scale)
    local diff = (output - target):squeeze()
    local _, h, w = table.unpack(diff:size():totable())
    local shave = scale + 6
    local diffShave = diff[{{}, {1 + shave, h - shave}, {1 + shave, w - shave}}]
    local psnr = -10 * math.log10(diffShave:pow(2):mean())

    return psnr
end

--in-place quantization and divide by 255
function util:quantize(img, mulImg)
    return img:mul(255 / mulImg):add(0.5):floor():div(255)
end

function util:swapModel(model, index)
    local sModel = nn.Sequential()

    if self.opt.netType == 'moresnet' and self.opt.mobranch < 1 then
        local sSeq = nn.Sequential()
        for i = 1, model:size() do
            local subModel = model:get(i)
            local modelName = subModel.__typename
            if modelName:find('Sequential') then
                local mainSeq = subModel:get(1):get(2)
                for j = 1, mainSeq:size() - 1 do
                    sSeq:add(mainSeq:get(j))
                end
                subSeq = mainSeq:get(mainSeq:size()):get(index)
                for j = 1, subSeq:size() do
                    sSeq:add(subSeq:get(j))
                end
                sModel:add(nn.ConcatTable()
                            :add(nn.Identity())
                            :add(sSeq))
                        :add(nn.CAddTable(true))
            elseif modelName:find('ParallelTable') then
                sModel:add(subModel:get(index))
            elseif not modelName:find('MultiSkipAdd') then
                sModel:add(subModel)
            end
        end
    else
        for i = 1, model:size() do
            local subModel = model:get(i)
            local modelName = subModel.__typename
            if modelName:find('ParallelTable') then
                sModel:add(subModel:get(index))
            elseif modelName:find('ConcatTable') then
                local isSkip = false
                for i = 1, subModel:size() do
                    if subModel:get(i).__typename:find('Identity') then
                        isSkip = true
                        break
                    end
                end
                if isSkip then
                    sModel:add(subModel)
                else
                    sModel:add(subModel:get(index))
                end
            else
                sModel:add(subModel)
                if modelName:find('MultiSkipAdd') then
                    sModel:add(nn.SelectTable(index))
                end
            end
        end
    end

    return sModel
end

function util:chopForward(input, model, scale)
    local b, c, h, w = unpack(input:size():totable())
    local shave = 30
    local sizeAvailable = 400 * 400
    
    if (h * w) < sizeAvailable then
        local output = model:forward(input):clone()
        model:clearState()
        collectgarbage()
        collectgarbage()
        return output
    end

    local wHalf1, hHalf1 = math.floor(w / 2), math.floor(h / 2)
    local wHalf2, hHalf2 = w - wHalf1, h - hHalf1
    local w1, w2 = wHalf1 + shave, w - wHalf2 - shave
    local h1, h2 = hHalf1 + shave, h - hHalf2 - shave

    local p1 = util:chopForward(input[{{}, {}, {1, h1}, {1, w1}}], model, scale)
    local p2 = util:chopForward(input[{{}, {}, {1, h1}, {w2 + 1, w}}], model, scale)
    local p3 = util:chopForward(input[{{}, {}, {h2 + 1, h}, {1, w1}}], model, scale)
    local p4 = util:chopForward(input[{{}, {}, {h2 + 1, h}, {w2 + 1, w}}], model, scale)
    local ret = torch.CudaTensor(b, c, scale * h, scale * w)
    w, h = scale * w, scale * h
    w1, w2, h1, h2 = scale * w1, scale * w2, scale * h1, scale * h2
    wHalf1, wHalf2, hHalf1, hHalf2 = scale * wHalf1, scale * wHalf2, scale * hHalf1, scale * hHalf2

    ret[{{}, {}, {1, hHalf1}, {1, wHalf1}}] = p1[{{}, {}, {1, hHalf1}, {1, wHalf1}}]
    ret[{{}, {}, {1, hHalf1}, {wHalf1 + 1, w}}] = p2[{{}, {}, {1, hHalf1}, {wHalf1 - w2 + 1, w - w2}}]
    ret[{{}, {}, {hHalf1 + 1, h}, {1, wHalf1}}] = p3[{{}, {}, {hHalf1 - h2 + 1, h - h2}, {1, wHalf1}}]
    ret[{{}, {}, {hHalf1 + 1, h}, {wHalf1 + 1, w}}] = p4[{{}, {}, {hHalf1 - h2 + 1, h - h2}, {wHalf1 - w2 + 1, w - w2}}]

    return ret
end

function util:recursiveForward(input, model, safe)
    model:clearState()
    local input = input:clone()
    local model = model:clone()
    collectgarbage()
    collectgarbage()

    local gpuid = self.opt and self.opt.gpuid or 1

    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    local function _recursion(input, subModel)
        subModel:clearState()
        collectgarbage()
        collectgarbage()

        local output
        local free, total = cutorch.getMemoryUsage(gpuid)

        if subModel.__typename:find('Sequential') then
            output = input
            for i = 1, #subModel do
                output = _recursion(output, subModel:get(i))
            end

        elseif subModel.__typename:find('ConcatTable') then
            output = {}
            for i = 1, subModel:size() do 
                local _output = _recursion(input, subModel:get(i))
                table.insert(output, _output)

                _output = nil
                subModel:clearState()
                collectgarbage()
                collectgarbage()
            end

        elseif subModel.__typename:find('FlattenTable') then
            output = input[self.opt.selOut]:clone() --choose output which you want

        elseif subModel.__typename:find('Concat') then  -- nn.Concat layer
            output = {}
            for i = 1, subModel:size() do
                local _output = _recursion(input, subModel:get(i))
                table.insert(output, _output)

                _output = nil
                subModel:clearState()
                collectgarbage()
                collectgarbage()
            end
            output = torch.cat(output, subModel.dimension)

        elseif subModel.__typename:find('Convolution') then
            local subModel = subModel:clone():cuda()
            collectgarbage()
            collectgarbage()

            assert(input:dim() == 4, 'Input dimension should be 4')
            local nInputPlane, nOutputPlane = subModel.nInputPlane, subModel.nOutputPlane
            local kH,kW, dH,dW = subModel.kH, subModel.kW, subModel.dH, subModel.dW
            local padH, padW = subModel.padH, subModel.padW
            local oH, oW
            if subModel.__typename:find('SpatialConvolution') then
                oH = math.floor((input:size(3) + 2*padH - kH) / dH + 1)
                oW = math.floor((input:size(4) + 2*padW - kW) / dW + 1)
            elseif subModel.__typename:find('SpatialFullConvolution') then
                oH = (input:size(3) - 1) * dH - 2 * padH + kH + subModel.adjH
                oW = (input:size(4) - 1) * dW - 2 * padW + kW + subModel.adjW
            end

            local nOutputPixel = nOutputPlane * oH * oW
            local failed = false

            if input:type() == 'torch.CudaTensor' then
                if 4 * nOutputPixel < free then
                    local _output
                    if pcall(function() _output = subModel:forward(input) end) then
                        if 4 * 2 * nOutputPixel < free then
                            if pcall(function() output = _output:clone() end) then
                                _output = nil
                            else
                                output = _output:float():clone()
                            end
                        else
                            output = _output:float():clone()
                        end
                    else
                        failed = true
                    end
                    _output = nil
                    subModel:clearState()
                    collectgarbage()
                    collectgarbage()
                end

                -- If input and output cannot reside in the memory at the same time
                if failed or 4 * nOutputPixel >= free then
                    local floatOutput = torch.Tensor(1, nOutputPlane, oH, oW)
                    local idx = 0

                    local splitSize = math.min(math.floor(free / (4 * oH * oW)), nOutputPlane)
                    local splitOutput = torch.CudaTensor()
                    while true do
                        if pcall(function() splitOutput:resize(1, splitSize, oH, oW) end) then
                            break
                        else
                            splitSize = math.floor(splitSize / 2)
                            splitOutput = torch.CudaTensor()
                            collectgarbage()
                            collectgarbage()
                        end

                        if splitSize < 4 then
                            error('Too low split size!')
                        end
                    end

                    local conv
                    while idx < nOutputPlane do
                        local split = math.min(nOutputPlane - idx, splitSize)
                        subModel:float()
                        collectgarbage()
                        collectgarbage()
                        if subModel.__typename:find('SpatialConvolution') then
                            conv = cudnn.SpatialConvolution(nInputPlane, split, kH, kW, dH, dW, padH, padW)
                            conv.weight:copy(subModel.weight[{{idx + 1, idx + split}}])
                            conv.bias:copy(subModel.bias[{{idx + 1, idx + split}}])
                        elseif subModel.__typename:find('SpatialFullConvolution') then
                            local adjH, adjW = subModel.adjH, subModel.adjW
                            conv = cudnn.SpatialFullConvolution(nInputPlane, split, kH, kW, dH, dW, padH, padW, adjH, adjW)
                            conv.weight:copy(subModel.weight[{{},{idx + 1, idx + split}}])
                            conv.bias:copy(subModel.bias[{{idx + 1, idx + split}}])
                        end
                        conv:cuda()

                        conv.output = splitOutput[{{},{1,split}}]

                        conv:forward(input)
                        floatOutput[{{},{idx + 1, idx + split}}]:copy(conv.output:float())

                        idx = idx + split
                    end

                    output = floatOutput:clone()
                    floatOutput = nil
                    splitOutput = nil
                    conv:clearState()
                    conv = nil
                end

            elseif input:type() == 'torch.FloatTensor' then -- this is the worst case
                local _, ch, h, w = table.unpack(input:size():totable())
                local splitSizeInput = ch / 2
                local splitInput = torch.CudaTensor()
                while true do
                    if pcall(function() splitInput:resizeAs(input[{{},{1,splitSizeInput}}]) end) then
                        break
                    else
                        splitSizeInput = splitSizeInput / 2
                        splitInput = torch.CudaTensor()
                        collectgarbage()
                        collectgarbage()
                    end

                    if splitSizeInput < 4 then
                        error('Too low split size!')
                    end
                end

                local _free = cutorch.getMemoryUsage(gpuid)
                local splitSizeOutput = math.min(math.floor(_free / (4 * oH * oW)), nOutputPlane)
                local splitOutput = torch.CudaTensor()
                while true do
                    if pcall(function() splitOutput:resizeAs(1, splitSizeOutput, oH, oW) end) then
                        break
                    else
                        splitSizeOutput = splitSizeOutput / 2
                        splitOutput = torch.CudaTensor()
                        collectgarbage()
                        collectgarbage()
                    end

                    if splitSizeOutput < 4 then
                        error('Too low split size!')
                    end
                end

                local idxInput, idxOutput = 0
                local floatOutput = torch.Tensor(1, nOutputPlane, oH, oW):zero()

                subModel:float()
                collectgarbage()
                collectgarbage()

                local conv
                while idxInput < nInputPlane do
                    local splitSizeInput = math.min(nInputPlane - idxInput, splitSizeInput)    
                    splitInput:resize(1, splitSizeInput, oH, oW)
                    splitInput:copy(input[{{},{idxInput + 1, idxInput + splitSizeInput}}])
                    while idxOutput < nOutputPlane do
                        local splitSizeOutput = math.min(nOutputPlane - idxOutput, splitSizeOutput)

                        if subModel.__typename:find('SpatialConvolution') then
                            conv = cudnn.SpatialConvolution(splitSizeInput, splitSizeOutput, kH, kW, dH, dW, padH, padW):noBias()
                            conv.weight:copy(subModel.weight[{{idxOutput + 1, idxOutput + splitSizeOutput},{idxInput + 1, idxInput + splitSizeInput}}])
                        elseif subModel.__typename:find('SpatialFullConvolution') then
                            local adjH, adjW = subModel.adjH, subModel.adjW
                            conv = cudnn.SpatialFullConvolution(splitSizeInput, splitSizeOutput, kH, kW, dH, dW, padH, padW, adjH, adjW):noBias()
                            conv.weight:copy(subModel.weight[{{idxInput + 1, idxInput + splitSizeInput},{idxOutput + 1, idxOutput + splitSizeOutput}}])
                        end
                        conv:cuda()

                        conv.output = splitOutput[{{},{1, splitSizeOutput}}]

                        conv:forward(splitInput)
                        floatOutput[{{},{idxOutput + 1, idxOutput + splitSizeOutput}}]:add(conv.output:float())

                        idxOutput = idxOutput + splitSizeOutput
                    end
                    idxInput = idxInput + splitSizeInput
                end
                floatOutput:add(subModel.bias:view(1,subModel.bias:size(1),1,1):repeatTensor(1,1,oH,oW))

                output = floatOutput:clone()
                floatOutput = nil
                splitInput = nil
                splitOutput = nil
                conv:clearState()
                conv = nil
            else
                error('Unknown input datatype')
            end

            subModel:clearState()
            subModel = nil
            collectgarbage()
            collectgarbage()

        elseif subModel.__typename:find('Shuffle') then
            local subModel = subModel:clone():cuda()
            collectgarbage()
            collectgarbage()

            assert(input:dim() == 4, 'Input dimension should be 4')
            local sc = subModel.upscaleFactor
            local nOutputPixel = input:numel() * sc * sc
            if 4 * 2 * nOutputPixel < free then
                output = subModel:forward(input):clone()
            elseif 4 * nOutputPixel < free then
                output = subModel:forward(input)
                output = output:float():clone()
            else
                local _, ch, h, w = table.unpack(input:size():totable())
                local nInputPlane, nOutputPlane = ch, ch / (sc * sc)
                local floatOutput = torch.Tensor(1, nOutputPlane, h * sc, w * sc)
                local idx = 0
                local splitSize = math.min(
                    math.floor((0.9 * free / (4 * h * w)) / (sc * sc)) * (sc * sc),
                    nInputPlane)
                
                while idx < nInputPlane do
                    local splitSizeInput = math.min(nInputPlane - idx, splitSize)
                    local splitSizeOutput = splitSizeInput / (sc * sc)
                    local splitInput = input[{{},{idx + 1, idx + splitSizeInput}}]
                    local splitOutput = subModel:forward(splitInput):float():clone()
                    local idxOutput = idx / (sc * sc)
                    floatOutput[{{},{idxOutput + 1, idxOutput + splitSizeOutput}}]:copy(splitOutput)

                    subModel:clearState()
                    splitOutput = nil
                    splitInput = nil
                    collectgarbage()
                    collectgarbage()

                    idx = idx + splitSizeInput
                end
                output = floatOutput:clone()
                floatOutput = nil
            end
        elseif subModel.__typename:find('ReLU') or subModel.__typename:find('MulConst') then
            local subModel = subModel:clone():cuda()
            collectgarbage()
            collectgarbage()

            assert(input:dim() == 4, 'Input dimension should be 4')

            local failed = false

            local _output
            if pcall(function() _output = subModel:forward(input) end) then
                if 4 * input:numel() < free then
                    if pcall(function() output = _output:clone() end) then
                        _output = nil
                    else
                        output = _output:float():clone()
                    end
                else
                    output = _output:float():clone()
                end
                _output = nil
            else
                local _, ch, h, w = table.unpack(input:size():totable())
                local floatOutput = torch.FloatTensor(input:size())
                local idx = 0

                local splitSize = math.min(math.floor(free / (4 * h * w)), input:size(2))
                local splitOutput = torch.CudaTensor()
                while true do
                    if pcall(function() splitOutput:resize(1, splitSize, oH, oW) end) then
                        break
                    else
                        splitSize = math.floor(splitSize / 2)
                        splitOutput = torch.CudaTensor()
                        collectgarbage()
                        collectgarbage()
                    end

                    if splitSize < 4 then
                        error('Too low split size!')
                    end
                end

                while idx < input:size(2) do
                    local splitSizeInput = math.min(input:size(2) - idx, splitSize)
                    local splitInput = input[{{},{idx + 1, idx + splitSizeInput}}]

                    subModel.output = splitOutput[{{},{1,splitSizeInput}}]
                    subModel:forward(splitInput)

                    floatOutput[{{},{idx + 1, idx + splitSizeInput}}]:copy(subModel.output:float())

                    idx = idx + splitSizeInput
                end

                output = floatOutput:clone()
                floatOutput = nil
                splitOutput = nil
                splitInput = nil
            end
            subModel:clearState()
            subModel = nil
            collectgarbage()
            collectgarbage()
        elseif subModel.__typename:find('FlattenTable') then
            output = subModel:forward(input)
        elseif subModel.__typename:find('SelectTable') then
            output = subModel:forward(input)
        elseif subModel.__typename:find('NarrowTable') then
            output = subModel:forward(input)
        elseif subModel.__typename:find('MultiSkipAdd') then
            output = subModel:forward(input)
        elseif subModel.__typename:find('ParallelTable') then
            output = {}
            for i = 1, #input do
                table.insert(output, subModel:get(i):forward(input[i]):clone())
            end
        elseif subModel.__typename:find('Identity') then
            output = input

        else -- What else? Please add other modules manually
            local subModel = subModel:clone():cuda()
            collectgarbage()
            collectgarbage()

            local _output
            if pcall(function() _output = subModel:forward(input) end) then
                if pcall(function() output = _output:clone() end) then
                    _output = nil
                else
                    output = _output:float():clone()
                end
                _output = nil
                subModel:clearState()
                collectgarbage()
                collectgarbage()
            else
                print('Please handle this layer in recursiveForward function: ')
                print(subModel)
                input_ = input
                subModel_ = subModel
                require 'trepl'()
            end

            subModel:clearState()
            subModel = nil
            collectgarbage()
            collectgarbage()
        end

        input = nil
        subModel:clearState()
        subModel = nil
        model:clearState()
        collectgarbage()
        collectgarbage()

        local function recursiveCuda(elem)
            if type(elem) == 'table' then
                for k,v in pairs(elem) do
                    v = recursiveCuda(v)
                end
                return elem
            elseif type(elem) == 'userdata' then
                if elem:type() == 'torch.CudaTensor' then
                    return elem
                else
                    -- if not pcall(function() elem = elem:cuda() end) then
                    --     ee = elem
                    --     _fr = cutorch.getMemoryUsage(1)
                    --     print('elem: ' .. elem:numel() * 4 / 1e9)
                    --     print('free: ' .. _fr / 1e9)
                    --     require 'trepl'()
                    -- end
                    pcall(function() elem = elem:cuda() end)
                    collectgarbage()
                    collectgarbage()
                    return elem
                end
            end
        end

        output = recursiveCuda(output)

        return output
    end

    local _ret = _recursion(input, model)
    local ret = _ret:float()

    _ret = nil
    model:clearState()
    model = nil
    input = nil
    collectgarbage()
    collectgarbage()

    return ret:cuda()
end

function util:x8Forward(img, model, scale)
    local function _rot90k(_img, k)
        k = (k + 4) % 4
        local _c, _h, _w = table.unpack(_img:size():totable())
        local ml = math.max(_h, _w)
        local buffer = torch.Tensor(_c, ml, ml)
        local hMargin = ml - _h
        local wMargin = ml - _w
        buffer[{{}, {1, _h}, {1, _w}}] = _img
        buffer = image.rotate(buffer, k * math.pi / 2)
        
        if _w > _h then
            if k == 1 then
                return buffer[{{}, {1, _w}, {1, _h}}]
            elseif k == 3 then
                return buffer[{{}, {1, _w}, {_w - _h + 1, _w}}]
            end
        else
            if k == 1 then
                return buffer[{{}, {_h - _w + 1, _h}, {1, _h}}]
            elseif k == 3 then
                return buffer[{{}, {1, _w}, {1, _h}}]
            end
        end
    end
    
    local us = nn.Unsqueeze(1):cuda()
    local output = util:chopForward(us:forward(img:cuda()), model, scale):squeeze(1)
    for j = 0, 7 do
        if j ~= 0 then
            local jmod4 = j % 4
            local augInput = img
            if j > 3 then
                augInput = image.hflip(augInput)
            end
            if jmod4 == 2 then
                augInput = image.rotate(augInput, jmod4 * math.pi / 2)
            elseif (jmod4 == 1) or (jmod4 == 3) then
                augInput = _rot90k(augInput, jmod4)
            end
            
            local augOutput = util:chopForward(us:forward(augInput:cuda()), model, scale):squeeze(1):float()

            if jmod4 == 2 then
                augOutput = image.rotate(augOutput, -jmod4 * math.pi / 2)
            elseif (jmod4 == 1) or (jmod4 == 3) then
                augOutput = _rot90k(augOutput, -jmod4)
            end
            if j > 3 then
                augOutput = image.hflip(augOutput)
            end
            output:add(augOutput:cuda())
        end
    end
    output:div(8)

    return output
end

return M.util
