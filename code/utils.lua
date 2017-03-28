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

function util:plot(tb,name)
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
    else
        table.insert(lines, {name, torch.Tensor(tb), '-'})
    end
    gnuplot.plot(lines)
    gnuplot.grid(true)
    gnuplot.title(name)
    gnuplot.xlabel('iteration (*' .. self.opt.testEvery .. ')')
    if torch.type(tb[1]):find('Tensor') then
        if tb[1][1] < tb[#tb][1] then
            gnuplot.movelegend('right', 'bottom')
        else
            gnuplot.movelegend('right', 'top')
        end
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
