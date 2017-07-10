require 'nn'
require 'cunn'
require 'cudnn'

local VGG_loss, parent = torch.class('nn.VGG_loss','nn.Criterion')

function VGG_loss:__init(opt)
    local conv_block = tonumber(opt.vggDepth:sub(1,1))
    local conv_layer = tonumber(opt.vggDepth:sub(3,3))
    local conv_cnt, pool_cnt = 0,0
    local layer_cut = 0
    if not paths.filep('../VGG-19_truncated.t7') then
        os.execute('wget -c http://cv.snu.ac.kr/research/EDSR/VGG-19_truncated.tar')
        os.execute('tar -xvf ../VGG-19_truncated.tar -C ../')
    end
    local vgg_19 = torch.load('../VGG-19_truncated.t7')
    for i=1,#vgg_19 do
        local layer_name = tostring(vgg_19:get(i)):lower()
        if layer_name:find('conv') then
            conv_cnt = conv_cnt + 1
        elseif layer_name:find('pool') then
            pool_cnt = pool_cnt + 1
            conv_cnt = 0
        end
        if pool_cnt == conv_block-1 and conv_cnt == conv_layer then
            layer_cut = i
            break
        end
    end

    local vgg = nn.Sequential()
    for i=1,layer_cut do vgg:add(vgg_19:get(i):clone()) end
        local RGB2BGR = nn.SpatialConvolution(3,3,1,1):noBias()
        RGB2BGR.weight = torch.Tensor({{0,0,1},{0,1,0},{1,0,0}})
    vgg:insert(RGB2BGR, 1)
        local mean = torch.Tensor({103.939 / 255, 116.779 / 255, 123.68 / 255}):mul(opt.mulImg)
        local subMean = nn.SpatialConvolution(3,3,1,1)
        subMean.weight = torch.eye(3,3):view(3,3,1,1)
        subMean.bias = torch.Tensor(mean):mul(-1)
    vgg:insert(subMean, 2)

    if opt.nChannel == 1 then
        vgg:insert(nn.Replicate(3), 1)
        vgg:insert(nn.SplitTable(1), 2)
        vgg:insert(nn.JoinTable(1,3), 3)
    end

    for i=1,#vgg do
        vgg:get(i).accGradParameters = function(input,gradOutput,scale) return end
    end
--[[
    if opt.nGPU > 1 then
        local gpus = torch.range(1, opt.nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
            :add(vgg, gpus)
            :threads(function()
                local cudnn = require 'cudnn'
                cudnn.fastest, cudnn.benchmark = fastest, benchmark
            end)
        dpt.gradInput = nil

        vgg = dpt:cuda()
    end
--]]

    self.vgg = vgg
    self.crit = nn.MSECriterion()

    if opt.nGPU > 0 then
        if opt.backend == 'cudnn' then
            self.vgg = cudnn.convert(self.vgg,cudnn)
        end
        self.vgg:cuda()
        self.crit:cuda()
    end

    self.opt = opt
end

function VGG_loss:updateOutput(input,target)
--[[
    self:copyInputs(input,target)

    self.vgg_target = self.vgg:forward(self.target):clone()
    self.vgg_input = self.vgg:forward(self.input):clone()
--]]
    self.vgg_target = self.vgg:forward(target):clone()
    self.vgg_input = self.vgg:forward(input):clone()
    self.output = self.crit:forward(self.vgg_input,self.vgg_target)

    return self.output
end

function VGG_loss:updateGradInput(input,target)

    self.dl_do = self.crit:backward(self.vgg_input,self.vgg_target)
--[[
    if self.opt.nGPU > 1 then
        self.tmp = self.tmp or cutorch.createCudaHostTensor()
        self.tmp:resize(self.dl_do:size()):copy(self.dl_do)
        self.dl_do = self.tmp
    end
    local vgg = self.vgg
    if torch.type(self.vgg)=='nn.DataParallelTable' then vgg = vgg:get(1) end
    self.gradInput = vgg:updateGradInput(self.input,self.dl_do)
--]]

    self.gradInput = self.vgg:updateGradInput(input,self.dl_do)

    return self.gradInput
end

function VGG_loss:copyInputs(input,target)
    self.input = self.input or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
    self.target = self.target or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())

    self.input:resize(input:size()):copy(input)
    self.target:resize(target:size()):copy(target)
end