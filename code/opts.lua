require 'cutorch'

local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    local now = os.date("%Y-%m-%d_%H-%M-%S")

    cmd:text()
    cmd:text('NTIRE 2017 Super-Resolution Challage')
    cmd:text('Team SNU-CVLAB')
    cmd:text()
    cmd:text('Options:')
    -- Global
    cmd:option('-manualSeed',       0,          'Manually set RNG seed')
    cmd:option('-nGPU',             1,          'Number of GPUs to use by default')
    cmd:option('-gpuid',            1,          'GPU id to use')
    cmd:option('-nThreads',         3,          'number of data loading threads')
    cmd:option('-save',             now,        'subdirectory to save/log experiments in')
    -- Data
    cmd:option('-datadir',          '/var/tmp',  'dataset location')
    cmd:option('-dataset',          'div2k',    'dataset for training: div2k | imagenet')
    cmd:option('-datatype',         't7',       'dataset type: png | t7')
    cmd:option('-dataSize',         'small',    'input image size: small | big')
    cmd:option('-degrade',          'bicubic',  'degrade type: bicubic | unknwon')
    cmd:option('-numVal',           10,         'number of images for validation')
    cmd:option('-colorAug',         'false',    'apply color augmentation (brightness, contrast, saturation')
    cmd:option('-subMean',          'false',     'data pre-processing: subtract mean')
    cmd:option('-divStd',           'false',     'data pre-processing: subtract mean and divide std')
    cmd:option('-mulImg',           255,          'data pre-processing: multiply constant value to both input and output')
    -- Training
    cmd:option('-nEpochs',          0,          'Number of total epochs to run. 0: Infinite')
    cmd:option('-startEpoch',       0,          'Manual epoch number for resuming the training. Default is the end')
    cmd:option('-manualDecay',      1e8,        'Reduce the learning rate by half per n epoch')
    cmd:option('-batchSize',        32,         'mini-batch size (1 = pure stochastic)')
    cmd:option('-patchSize',        96,         'Training patch size')
    cmd:option('-scale',            2,          'Super-resolution upscale factor')
    cmd:option('-testOnly',         'false',     'Run on validation set only')
    cmd:option('-printEvery',       1e2,        'Print log every # iterations')
    cmd:option('-testEvery',        1e3,        'Test every # iterations')
    cmd:option('-load',             '.',        'Load saved training model, history, etc.')
    cmd:option('-clip',             -1,         'Gradient clipping constant(theta)')
    -- Optimization
    cmd:option('-optimMethod',      'ADAM',     'Optimization method')
    cmd:option('-lr',               1e-4,       'initial learning rate')
    cmd:option('-lrLow',            1,          'Relative learning rate of low frequency components')
    cmd:option('-lrHigh',           1,          'Relative learning rate of high frequency components')
    cmd:option('-momentum',         0.9,        'SGD momentum')
    cmd:option('-beta1',            0.9,        'ADAM beta1')
    cmd:option('-beta2',            0.999,      'ADAM beta2')
    cmd:option('-epsilon',          1e-8,       'ADAM epsilon')
    -- Model
    cmd:option('-netType',          'resnet',   'SR network architecture. Options: resnet | vdsr | msresnet')
    cmd:option('-filtsize',         3,          'Filter size of convolutional layer')
    cmd:option('-nLayer',           20,         'Number of convolution layer (for VDSR)')
    cmd:option('-nConv',            34,         'Number of convolution layers excluding the beginning and end')
    cmd:option('-nResBlock',        16,         'Number of residual blocks in SR network (for SRResNet, SRGAN)')
    cmd:option('-nChannel',         3,          'Number of input image channels: 1 or 3')
    cmd:option('-nFeat',            64,         'Number of feature maps in residual blocks in SR network')
    cmd:option('-upsample',         'shuffle',  'Upsampling method: full | bilinear | shuffle')
    cmd:option('-trainNormLayer',   'false',    'Train normalization layer')
    cmd:option('-selOut',           2,          'Select output if there exists multiple outputs in model')
    cmd:option('-modelVer',         1,          'Experimental model version')
    -- Loss
    cmd:option('-abs',              0,          'L1 loss weight')
    cmd:option('-chbn',             0,          'Charbonnier loss weight')
    cmd:option('-smoothL1',         0,          'Smooth L1 loss weight')
    cmd:option('-mse',              1,          'MSE loss weight')
    cmd:option('-ssim',             0,          'SSIM loss weight')
    cmd:option('-band',             0,          'Band loss weight')
    cmd:option('-grad',             0,          'Gradient loss weight')
    cmd:option('-grad2',            0,          '2nd order Gradient loss weight')
    cmd:option('-gradDist',         'mse',      'Distance of gradient loss')
    cmd:option('-gradPrior',        0,          'Gradient prior weight')
    cmd:option('-gradPower',        0.8,        'Gradient prior power')
    cmd:option('-fd',               0,          'Fourier loss weight')
    cmd:option('-fdwc',             0.75,       'Fourier loss wc')
    cmd:option('-fdFilter',         'lowpass',  'Fourier loss filter type')
    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.colorAug = opt.colorAug == 'true'
    opt.subMean = opt.subMean == 'true'
    opt.divStd = opt.divStd == 'true'
    opt.trainNormLayer = opt.trainNormLayer == 'true'
    opt.testOnly = opt.testOnly == 'true'

    if opt.load ~= '.' then 
        opt.save = opt.load
        if not paths.dirp(paths.concat('../experiment',opt.save)) then
            print(opt.load .. ' does not exist. Train new model.')
            opt.load = false
        end
    else
        opt.load = false
    end

    opt.save = paths.concat('../experiment',opt.save)
    if not paths.dirp(opt.save) then
        paths.mkdir(opt.save)
        paths.mkdir(paths.concat(opt.save,'result'))
        paths.mkdir(paths.concat(opt.save,'model'))
    end

    --torch.setnumthreads(1)
    torch.manualSeed(opt.manualSeed)
    torch.setdefaulttensortype('torch.FloatTensor')

    if opt.nGPU == 1 then
        cutorch.setDevice(opt.gpuid)
    end
    cutorch.manualSeedAll(opt.manualSeed)

    if (opt.nEpochs == 0) then
        opt.nEpochs = math.huge
    end

    opt.optimState = {
        learningRate = opt.lr,
        momentum = opt.momentum,
        dampening = 0,
        nesterov = true,
        beta1 = opt.beta1,
        beta2 = opt.beta2,
        epsilon = opt.epsilon
    }
    if opt.optimMethod == 'SGD' then 
        opt.optimState.method = optim.sgd
    elseif opt.optimMethod == 'ADADELTA' then
        opt.optimState.method = optim.adadelta
    elseif opt.optimMethod == 'ADAM' then
        opt.optimState.method = optim.adam
    elseif opt.optimMethod == 'RMSPROP' then
        opt.optimState.method = optim.rmsprop
    else
        error('unknown optimization method')
    end  

    local opt_text = io.open(paths.concat(opt.save,'options.txt'),'a')
    opt_text:write(os.date("%Y-%m-%d_%H-%M-%S\n"))
    local function save_opt_text(key,value)
        if type(value) == 'table' then
            for k,v in pairs(value) do
                save_opt_text(k,v)
            end
        else
            if type(value) == 'function' then
                value = 'function'
            elseif type(value) == 'boolean' then
                value = value and 'true' or 'false'
            elseif type(value) == 'userdata' then
                value = 'torch tensor'
            end
            opt_text:write(key .. ' : ' .. value .. '\n')
            return 
        end
    end
    save_opt_text(_,opt)
    opt_text:write('\n\n\n')
    opt_text:close()

    return opt
end

return M
