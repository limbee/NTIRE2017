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

function util:plot(tb, name, label)
    local fig = gnuplot.pdffigure(paths.concat(self.save, name .. '.pdf'))
    local lines = {}
    if torch.type(tb[1]):find('Tensor') then
        local nLine = tb[1]:size(1)
        local value = {}
        for i = 1, nLine do 
            value[i] = torch.Tensor(#tb)
            for j = 1, #tb do
                value[i][j] = tb[j][i]
            end
            table.insert(lines, {name .. ' x' .. tostring(i + 1) , value[i], '-'})
        end
        gnuplot.plot(lines)
    elseif type(tb[1]):find('table') then
        for i = 1, #tb do
            table.insert(lines, {label[i], torch.Tensor(tb[i]), '-'})
        end
        gnuplot.plot(table.unpack(lines))
    else
        table.insert(lines, {name, torch.Tensor(tb), '-'})
        gnuplot.plot(lines)
    end
    gnuplot.grid(true)
    gnuplot.title(name)
    gnuplot.xlabel('iteration (*' .. self.opt.testEvery .. ')')
    if torch.type(tb[1]):find('Tensor') then
        if tb[1][1] < tb[#tb][1] then
            gnuplot.movelegend('right', 'bottom')
        else
            gnuplot.movelegend('right', 'top')
        end
    elseif type(tb[1]):find('table') then
        gnuplot.movelegend('right', 'bottom')
    else
        if tb[1] < tb[#tb] then
            gnuplot.movelegend('right', 'bottom')
        else
            gnuplot.movelegend('right', 'top')
        end
    end
	gnuplot.plotflush(fig)
	gnuplot.closeall()  
end

function util:checkpoint(model, criterion, loss, psnr)
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    model:clearState()

    torch.save(paths.concat(self.save, 'model', 'model_' .. #loss .. '.t7'), model)
    torch.save(paths.concat(self.save, 'loss.t7'), loss)
    torch.save(paths.concat(self.save, 'psnr.t7'), psnr)
    torch.save(paths.concat(self.save, 'opt.t7'), self.opt)
end

function util:load()
    local ok, loss, psnr
    if self.opt.load then
        ok, loss, psnr, opt = pcall(
            function()
                local loss = torch.load(paths.concat(self.save, 'loss.t7'))
                local psnr = torch.load(paths.concat(self.save, 'psnr.t7'))
                local opt = torch.load(paths.concat(self.save, 'opt.t7'))
                return loss, psnr, opt
            end)
        if ok then
            print(('Loaded history (%d epoch * %d iter/epoch)\n'):format(#loss, self.opt.testEvery))
            if self.opt.startEpoch > #loss + 1 then
                error(('Start epoch cannot be bigger than history (%d epochs)'):format(#loss))
            elseif self.opt.startEpoch == 1 then
                error('Please set -startEpoch bigger than 1, if you want to resume the training')
            elseif self.opt.startEpoch > 1 and self.opt.startEpoch <= #loss then
                print(('Resuming the training from %d epoch'):format(self.opt.startEpoch))
                local _loss, _psnr = {}, {}
                for i = 1, self.opt.startEpoch - 1 do
                    _loss[i] = loss[i]
                    _psnr[i] = psnr[i]
                end
                loss = _loss
                psnr = _psnr
            else -- This is the default setting. startEpoch = 0 corresponds to #loss + 1
                print(('Continue training (%d epochs~)'):format(#loss + 1))
                self.opt.startEpoch = #loss + 1
            end
        else
            error('history (loss, psnr, options) does not exist')
        end
    else
        ok = false
        loss, psnr = {}, {}
        for i = 1, #self.opt.scale do
            table.insert(psnr, {})
        end
        self.opt.startEpoch = 1
    end

    if ok then
        local prevlr = self.opt.optimState.learningRate
        self.opt.optimState.learningRate = prevlr / math.pow(2, math.floor((#loss + 1) / self.opt.manualDecay))
        if self.opt.optimState.learningRate ~= prevlr then
            print(string.format('Learning rate decreased: %.6f -> %.6f',
            prevlr, self.opt.optimState.learningRate))
        end
    end

    return ok, loss, psnr
end

function util:calcPSNR(output, target, scale)
    local diff = (output - target):squeeze()
    local _, h, w = table.unpack(diff:size():totable())
    local shave = scale + 6
    --local diffShave = image.crop(diff, 'c', w - (2 * shave), h - (2 * shave))
    local diffShave = diff[{{}, {1 + shave, h - shave}, {1 + shave, w - shave}}]
    local psnr = -10 * math.log10(diffShave:pow(2):mean())

    return psnr
end

--in-place quantizing and divide by 255
function util:quantize(img, mulImg)
    return img:mul(255 / mulImg):add(0.5):floor():div(255)
end

function util:selectMultiOutput(model, index)
    local selectModel = nn.Sequential()
    for i = 1, model:size() do
        local modelName = model:get(i).__typename
        if modelName:find('ParallelTable') or modelName:find('ConcatTable') then
            selectModel:add(model:get(i):get(index))
        else
            selectModel:add(model:get(i))
        end
    end

    return selectModel, model
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
