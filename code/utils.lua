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

function util:plot(tbl, title, name)
    -- Assume tbl is a table of numbers, or a table of tables
    local fig = gnuplot.pdffigure(paths.concat(self.save, title .. '.pdf'))
    local name = name or title

    local function findMinMax(tb)
        local minKey, maxKey = math.huge, -math.huge    
        local minKeyValue, maxKeyValue
        for k, v in pairs(tb) do
            if k < minKey then 
                minKey = k
                minKeyValue = v
            end
            if k > maxKey then
                maxKey = k
                maxKeyValue = v
            end
        end
        return minKeyValue, maxKeyValue
    end

    local function typeTable(tb)
        for k, v in pairs(tb) do
            if type(v) == 'table' then
                return 'table'
            else
                return 'number'
            end
        end
    end

    local function toTensor(tb)
        local numel = 0
        for k,v in pairs(tb) do numel = numel + 1 end
        local ts = torch.Tensor(numel,2)
        local idx = 1
        for k, v in pairs(tb) do
            ts[idx][1] = k
            ts[idx][2] = v
        end
        return ts
    end

    local lines = {}
    local first, last
    if typeTable(tbl) ~= 'table' then -- single graph
        table.insert(lines, {name, toTensor(tbl), '-'})
        first, last = findMinMax(tbl)
    else -- multiple lines
        assert(type(name) == 'table', 'name must be a table, if you want to draw lines more than 1')
        for i = 1, #tbl do
            table.insert(lines, {name[i], toTensor(tbl[i]), '-'})
        end
        first, last = findMinMax(tbl[1])
    end

    if first < last then
        gnuplot.movelegend('right', 'bottom')
    else
        gnuplot.movelegend('right', 'top')
    end

    if not pcall(function() gnuplot.plot(lines) end) then
        ll = lines
        tt = tbl
        nn = name
        require 'trepl'()
    end
    gnuplot.grid(true)
    gnuplot.title(name)
    gnuplot.xlabel('iteration (*' .. self.opt.testEvery .. ')')
	gnuplot.plotflush(fig)
	gnuplot.closeall()  
end

function util:checkpoint(model, criterion, loss, psnr)
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    local numLoss = 0
    for k,v in pairs(loss) do numLoss = numLoss + 1 end

    torch.save(paths.concat(self.save, 'model', 'model_' .. numLoss .. '.t7'), model:clearState())
    torch.save(paths.concat(self.save, 'loss.t7'), loss)
    torch.save(paths.concat(self.save, 'psnr.t7'), psnr)
    torch.save(paths.concat(self.save, 'opt.t7'), self.opt)
end

function util:load()
    local ok, loss, psnr
    local numLoss = 0

    if self.opt.load then
        ok, loss, psnr, opt = pcall(
            function()
                local loss = torch.load(paths.concat(self.save, 'loss.t7'))
                local psnr = torch.load(paths.concat(self.save, 'psnr.t7'))
                local opt = torch.load(paths.concat(self.save, 'opt.t7'))
                return loss, psnr, opt
            end)
        if ok then
            local _lastIter = 0
            for k, v in pairs(loss) do
                if k > _lastIter then 
                    _lastIter = k
                end
                numLoss = numLoss + 1
            end
            print(('Loaded history (%d epochs = %d iterations)\n'):format(numLoss, _lastIter))
            if self.opt.startEpoch > numLoss + 1 then
                error(('Start epoch cannot be bigger than history (%d epochs)'):format(numLoss))
            elseif self.opt.startEpoch == 1 then
                error('Please set -startEpoch bigger than 1, if you want to resume the training')
            elseif self.opt.startEpoch > 1 and self.opt.startEpoch <= numLoss then
                print(('Resuming the training from %d epoch'):format(self.opt.startEpoch))
                local keys = {}
                for k, v in pairs(loss) do
                    tabls.insert(keys, k)
                end
                table.sort(keys)
                local lastIter = keys[self.opt.startEpoch - 1]

                local _loss, _psnr = {}, {}
                for k, v in pairs(loss) do
                    if k <= lastIter then
                        _loss[k] = loss[k]
                        _psnr[k] = psnr[k]
                    end
                end
                loss, psnr = _loss, _psnr
                self.opt.lastIter = lastIter
            else -- This is the default setting. startEpoch = 0 corresponds to #loss + 1
                print(('Continue training (After %d epochs = %d iterations)'):format(numLoss, _lastIter))
                self.opt.startEpoch = numLoss + 1
                self.opt.lastIter = _lastIter
            end
        else
            error('history (loss, psnr, options) does not exist')
        end
    else
        ok = false
        loss, psnr = {}, {}
        self.opt.startEpoch = 1
        self.opt.lastIter = 0
    end

    if ok then
        local prevlr = self.opt.optimState.learningRate
        self.opt.optimState.learningRate = prevlr / math.pow(2, math.floor((numLoss + 1) / self.opt.manualDecay))
        if self.opt.optimState.learningRate ~= prevlr then
            print(string.format('Learning rate decreased: %.6f -> %.6f',
            prevlr, self.opt.optimState.learningRate))
        end
    end

    return ok, loss, psnr
end

function util:calcPSNR(output,target,scale)
    output = output:squeeze()
    target = target:squeeze()
    local _,h,w = table.unpack(output:size():totable())
    local shave = scale + 6
    local diff = (output - target)[{{},{shave + 1, h - shave}, {shave + 1, w - shave}}]
    local mse = diff:pow(2):mean()
    local psnr = -10*math.log(mse,10)

    return psnr
end

--in-place quantizing and divide by 255
function util:quantize(img, mulImg)
    return img:mul(255 / mulImg):add(0.5):floor():div(255)
end

function util:recursiveForward(input, model)
    local __model = model:clone():clearState()
    if torch.type(model) == 'nn.DataParallelTable' then
        __model = __model:get(1)
    end

    local function _recursion(input, subModel)
        local output, gpuid
        if self.opt then
            gpuid = self.opt.gpuid
        else
            gpuid = 1
        end
        local free, total = cutorch.getMemoryUsage(gpuid)

        subModel:clearState() -- This is important, though I don't know the reason
        collectgarbage()
        collectgarbage()

        if subModel.__typename:find('ConcatTable') then
            output = {}
            for i = 1, subModel:size() do 
                table.insert(output, _recursion(input, subModel:get(i)))
            end
        elseif subModel.__typename:find('Concat') then  -- nn.Concat layer
            output = {}
            for i = 1, subModel:size() do
                table.insert(output, _recursion(input, subModel:get(i)))
            end
            output = torch.cat(output, subModel.dimension)
        elseif subModel.__typename:find('Sequential') then
            output = input
            for i = 1, #subModel do
                output = _recursion(output, subModel:get(i))
            end
        elseif subModel.__typename:find('Convolution') then
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
            if 4 * 2 * nOutputPixel < free then
                output = subModel:forward(input):clone()
            elseif 4 * nOutputPixel < free then
                output = subModel:forward(input)
                output = output:float():clone()
            else -- If input and output cannot reside in the memory at the same time
                local floatOutput = torch.Tensor(1, nOutputPlane, oH, oW)
                local idx = 0
                local splitSize = math.min(math.floor(0.9 * free / (4 * oH * oW)), nOutputPlane)
                while idx < nOutputPlane do
                    local split = math.min(nOutputPlane - idx, splitSize)
                    local conv
                    if subModel.__typename:find('SpatialConvolution') then
                        conv = nn.SpatialConvolution(nInputPlane, split, kH, kW, dH, dW, padH, padW)
                        conv.weight:copy(subModel.weight[{{idx + 1, idx + split}}])
                    elseif subModel.__typename:find('SpatialFullConvolution') then
                        local adjH, adjW = subModel.adjH, subModel.adjW
                        conv = nn.SpatialFullConvolution(nInputPlane, split, kH, kW, dH, dW, padH, padW, adjH, adjW)
                        conv.weight:copy(subModel.weight[{{},{idx + 1, idx + split}}])
                    end
                    conv.bias:copy(subModel.bias[{{idx + 1, idx + split}}])

                    conv = cudnn.convert(conv, cudnn)

                    local splitOutput = conv:cuda():forward(input):float():clone()
                    floatOutput[{{},{idx + 1, idx + split}}]:copy(splitOutput)

                    conv:clearState()
                    conv = nil
                    splitOutput = nil
                    collectgarbage()
                    collectgarbage()

                    idx = idx + split
                end
                output = floatOutput:clone()
                floatOutput = nil
            end
        elseif subModel.__typename:find('Shuffle') then
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
        elseif subModel.__typename:find('ReLU') then
            assert(input:dim() == 4, 'Input dimension should be 4')
            if 4 * input:numel() < free then
                output = subModel:forward(input):clone()
            else
                local _, ch, h, w = table.unpack(input:size():totable())
                local floatOutput = torch.FloatTensor(input:size())
                local idx = 0
                local splitSize = math.min(
                    math.floor(0.9 * free / (4 * h * w)),
                    input:size(2))

                while idx < input:size(2) do
                    local splitSizeInput = math.min(input:size(2) - idx, splitSize)
                    local splitInput = input[{{},{idx + 1, idx + splitSizeInput}}]
                    local splitOutput = subModel:forward(splitInput):float():clone()
                    floatOutput[{{},{idx + 1, idx + splitSizeInput}}]:copy(splitOutput)

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
        elseif subModel.__typename:find('FlattenTable') then
            output = input[self.opt.selOut] --choose output which you want
        elseif subModel.__typename:find('Identity') then
            output = input
        else -- What else? Please add other modules manually
            if not pcall(function() output = subModel:forward(input):clone() end) then
                output = nil
                subModel:clearState()
                collectgarbage()
                collectgarbage()
                if not pcall(function() output = subModel:forward(input):float():clone() end) then
                    print('Please handle this layer in recursiveForward function: ' .. subModel)
                    input_ = input
                    subModel_ = subModel
                    require 'trepl'()
                end
            end
        end

        input = nil
        __model:clearState()
        subModel:clearState()
        subModel = nil
        collectgarbage()
        collectgarbage()

        local function recursiveCuda(elem)
            if type(elem) == 'table' then
                for k,v in pairs(elem) do
                    v = recursiveCuda(v)
                end
                return elem
            elseif type(elem) == 'userdata' then
                elem = elem:cuda()
                return elem
            else
                ee = elem
                require 'trepl'()
            end
        end

        output = recursiveCuda(output)

        return output
    end

    local ret = _recursion(input, __model)
    __model = nil
    collectgarbage()
    collectgarbage()

    return ret
end

function util:x8Forward(img, model)
    local function _rot90k(_img, k)
        local _c, _h, _w = table.unpack(_img:size():totable())
        local ml = math.max(_h, _w)
        local buffer = torch.Tensor(_c, ml, ml)
        local hMargin = math.floor((ml - _h) / 2)
        local wMargin = math.floor((ml - _w) / 2)
        buffer[{{}, {1 + hMargin, ml - hMargin}, {1 + wMargin, ml - wMargin}}] = _img
        buffer = image.rotate(buffer, k * math.pi / 2)
        
        --return image.crop(buffer, 'c', ml - (2 * hMargin), ml - (2 * wMargin))
        return buffer[{{}, {1 + wMargin, ml - wMargin}, {1 + hMargin, ml - hMargin}}]
    end
    
    local us = nn.Unsqueeze(1):cuda()
    local output = util:recursiveForward(us:forward(img:cuda()), model):squeeze(1)
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
            
            local augOutput = util:recursiveForward(us:forward(augInput:cuda()), model):squeeze(1):float()

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
