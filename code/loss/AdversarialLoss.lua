require 'nn'
require 'cunn'
require 'cudnn'

local AdversarialLoss, parent = torch.class('nn.AdversarialLoss','nn.Criterion')

function AdversarialLoss:__init(opt)
    local loaded = false
    if opt.load then
        local modelPath = paths.concat(opt.save,'model','discriminator_latest.t7')
        if paths.filep(modelPath) then 
            print('Resuming discriminator from ' .. modelPath)
            local d = torch.load(modelPath)
            self.discriminator = torch.load(modelPath)
            loaded = true
        else
            print('Saved disriminator model not found in ' .. opt.save)
        end
    end

    if not loaded then
        local function conv_block(nInputPlane, nOutputPlane, filtsize, str)
            local pad = math.floor((filtsize-1)/2)
            local negval = opt.negval
            local block = nn.Sequential()
                :add(nn.SpatialConvolution(nInputPlane, nOutputPlane, filtsize, filtsize, str,str, pad,pad))
                :add(nn.LeakyReLU(negval, true))
                :add(nn.SpatialBatchNormalization(nOutputPlane))
            return block
        end

        local negval = opt.negval
        local filtsize_1 = 3
        local filtsize_2 = opt.filtsizeD
        local pad = (filtsize_1 - 1) / 2
        local ks = opt.patchSize / (2^4)

        local discriminator = nn.Sequential()
            :add(nn.SpatialConvolution(opt.nChannel,64,filtsize_1,filtsize_1,1,1,pad,pad))
            :add(nn.LeakyReLU(negval,true))
            :add(conv_block(64,64,filtsize_2,2))
            :add(conv_block(64,128,filtsize_1,1))
            :add(conv_block(128,128,filtsize_2,2))
            :add(conv_block(128,256,filtsize_1,1))
            :add(conv_block(256,256,filtsize_2,2))
            :add(conv_block(256,512,filtsize_1,1))
            :add(conv_block(512,512,filtsize_2,2))
            :add(nn.SpatialConvolution(512,1024,ks,ks)) -- dense.
            :add(nn.LeakyReLU(negval,true))
            :add(nn.SpatialConvolution(1024,1,1,1)) -- dense
            :add(nn.Sigmoid())

        self.discriminator = discriminator
    end
    self.crit = nn.BCECriterion()

    self.params, self.gradParams = self.discriminator:getParameters()
    self.feval = function() return self.err, self.gradParams end
    self.optimState = opt.optimState_D

    if opt.nGPU > 0 then
        if opt.backend == 'cudnn' then
            self.discriminator = cudnn.convert(self.discriminator,cudnn)
        end
        self.discriminator:cuda()
        self.crit:cuda()
    end
end

function AdversarialLoss:updateOutput(input,target,mode)

    self.d_output_fake = self.discriminator:forward(input):clone()
    self.d_target_real = self.d_output_fake.new():resizeAs(self.d_output_fake):fill(1)

    self.output = self.crit:forward(self.d_output_fake,self.d_target_real)

    return self.output
end

function AdversarialLoss:updateGradInput(input,target)
    self.gradOutput = self.crit:backward(self.d_output_fake,self.d_target_real):clone()
    self.gradInput = self.discriminator:updateGradInput(input,self.gradOutput):clone() -- return value

    -- discriminator train
    self.discriminator:zeroGradParameters()
        -- fake
    self.d_target_fake = self.d_output_fake.new():resizeAs(self.d_output_fake):fill(0)
    self.err_fake = self.crit:forward(self.d_output_fake,self.d_target_fake)
    local gradOutput_fake = self.crit:backward(self.d_output_fake,self.d_target_fake):clone()
    self.discriminator:backward(input,gradOutput_fake)
        -- real
    self.d_output_real = self.discriminator:forward(target):clone()
    self.err_real = self.crit:forward(self.d_output_real,self.d_target_real)
    local gradOutput_real = self.crit:backward(self.d_output_real,self.d_target_real):clone()
    self.discriminator:backward(target,gradOutput_real) 

    self.err = self.err_fake + self.err_real
    self.optimState.method(self.feval, self.params, self.optimState)

    return self.gradInput
end