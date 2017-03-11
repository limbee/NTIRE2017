require 'nn'
require 'cunn'
require 'cudnn'

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
    torch.save(paths.concat(self.save, 'model', 'model_latest.t7'), model)

    torch.save(paths.concat(self.save, 'loss.t7'), loss)
    torch.save(paths.concat(self.save, 'psnr.t7'), psnr)
    torch.save(paths.concat(self.save, 'opt.t7'), self.opt)
end

function util:load()
    local ok, loss, psnr
    if self.opt.load then
        ok, loss, psnr, opt = 
        pcall(function()
	        local loss = torch.load(paths.concat(self.save, 'loss.t7'))
            local psnr = torch.load(paths.concat(self.save, 'psnr.t7'))
            local opt = torch.load(paths.concat(self.save, 'opt.t7'))
            return loss, psnr, opt
        end)
        if ok then
            print(('loaded history (%d epoch * %d iter/epoch)\n'):format(#loss, self.opt.testEvery))
        else
            print('history (loss, psnr, options) does not exist')
            loss, psnr = {}, {}
        end
    else
        ok = false
        loss, psnr = {},{}
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
    local psnr = -10*math.log10(mse)

    return psnr
end

function util:recursiveForward(input, model)
    local __model = model:clone():clearState()
    if torch.type(model) == 'nn.DataParallelTable' then
        __model = __model:get(1)
    end

    local function _recursion(input, subModel)
        local output
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
        elseif subModel.__typename:find('Identity') then
            output = input:clone()
        else
            if subModel.__typename:find('Convolution') and subModel.nInputPlane + subModel.nOutputPlane > 256 then
                assert(input:dim() == 4, 'Input dimension should be 4')
                local splitSize, idx = 64, 0
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
                local floatOutput = torch.Tensor(1, nOutputPlane, oH, oW)

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

                    conv:clearState()
                    conv = nil
                    splitOutput = nil
                    collectgarbage()
                    collectgarbage()

                    idx = idx + split
                end
                output = floatOutput:cuda()
                floatOutput = nil
            elseif subModel.__typename:find('Shuffle') then
                local sc = subModel.upscaleFactor
                if input:size(2) * (1 + 1 / (sc * sc)) > 512 then
                    -- Then, input:size(2) should be (n^2)*m, where n is the parameter of PixelShuffle layer.
                    -- Restrict m < 64 at each forward
                    local nInputPlane, nOutputPlane = input:size(2), input:size(2) / (sc * sc)
                    local floatOutput = torch.Tensor(1, nOutputPlane, input:size(3) * sc, input:size(4) * sc)
                    local splitSize, idx = 256, 0
                    
                    while idx < nInputPlane do
                        local splitSizeInput = math.min(nInputPlane - idx, splitSize)
                        local splitSizeOutput = splitSizeInput / (sc * sc)
                        local splitInput = input[{{},{idx + 1, idx + splitSizeInput}}]                    
                        local splitOutput = subModel:forward(splitInput):clone():float()
                        local idxOutput = idx / (sc * sc)
                        floatOutput[{{},{idxOutput + 1, idxOutput + splitSizeOutput}}]:copy(splitOutput)

                        subModel:clearState()
                        splitOutput = nil
                        collectgarbage()
                        collectgarbage()

                        idx = idx + splitSizeInput
                    end
                    output = floatOutput:cuda()
                    floatOutput = nil
                end
            else
                output = subModel:forward(input):clone()
            end
        end
        input = nil
        subModel:clearState()
        subModel = nil
        __model:clearState()
        collectgarbage()
        collectgarbage()

        return output
    end

    local ret = _recursion(input, __model)
    __model = nil
    collectgarbage()
    collectgarbage()

    return ret
end

return M.util
