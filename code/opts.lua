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
    cmd:option('-manualSeed',   0,          'Manually set RNG seed')
    cmd:option('-nGPU',         1,          'Number of GPUs to use by default')
    cmd:option('-gpuid',        1,          'GPU id to use')
    cmd:option('-nThreads',     3,          'number of data loading threads')
    cmd:option('-save',         now,        'subdirectory to save/log experiments in')
    -- Data
    cmd:option('-dataset',      'div2k',    'dataset for training: div2k | imagenet')
    cmd:option('-datatype',     't7',       'dataset type: png | t7')
    cmd:option('-degrade',      'bicubic',  'degrade type: bicubic | unknwon')
    cmd:option('-numVal',       10,         'number of images for validation')
    -- Training
    cmd:option('-nEpochs',      0,          'Number of total epochs to run. 0: Infinite')
    cmd:option('-epochNumber',  1,          'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',    16,         'mini-batch size (1 = pure stochastic)')
    cmd:option('-patchSize',    96,         'Training patch size')
    cmd:option('-scale',        2,          'Super-resolution upscale factor')
    cmd:option('-testOnly',    'false',     'Run on validation set only')
    cmd:option('-printEvery',   1e2,        'Print log every # iterations')
    cmd:option('-testEvery',    1e3,        'Test every # iterations')
    cmd:option('-load',         '.',        'Load saved training model, history, etc.')
    cmd:option('-clip',         -1,         'Gradient clipping constant(theta)')
    -- Optimization
    cmd:option('-optimMethod',  'ADAM',     'Optimization method')
    cmd:option('-lr',           1e-4,       'initial learning rate')
    cmd:option('-lrLow',        1,          'Relative learning rate of low frequency components')
    cmd:option('-lrHigh',       2,          'Relative learning rate of high frequency components')
    cmd:option('-momentum',     0.9,        'SGD momentum')
    cmd:option('-beta1',        0.9,        'ADAM momentum')
    cmd:option('-weightDecay',  0,          'weight decay')
    -- Model
    cmd:option('-netType',      'bandnet',  'SR network architecture. Options: resnet | vdsr')
    cmd:option('-netwc',        0.2,        'Cut-off frequency of bandnet')
    cmd:option('-pre_act',      'false',    'Pre-activation architecture (for ResNet)')
    cmd:option('-bottleneck',   'false',    'Use bottleneck architecture (for ResNet and preResNet)')
    cmd:option('-nLayer',       20,         'Number of convolution layer (for VDSR)')
    cmd:option('-nResBlock',    16,         'Number of residual blocks in SR network (for SRResNet, SRGAN)')
    cmd:option('-nChannel',     3,          'Number of input image channels: 1 or 3')
    cmd:option('-nFeat',        64,         'Number of feature maps in residual blocks in SR network')
    cmd:option('-upsample',     'shuffle',  'Upsampling method: full | bilinear | shuffle')
    -- Loss
    cmd:option('-abs',          0,          'L1 loss weight')
    cmd:option('-chbn',         0,          'Charbonnier loss weight')
    cmd:option('-smoothL1',     0,          'Smooth L1 loss weight')
    cmd:option('-mse',          0,          'MSE loss weight')
    cmd:option('-ssim',         0,          'SSIM loss weight')
    cmd:option('-band',         1,          'Band loss weight')
    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.pre_act = opt.pre_act=='true'
    opt.bottleneck = opt.bottleneck=='true'

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
        os.execute('export CUDA_VISIBLE_DEVICES=' .. (opt.gpuid - 1))
        cutorch.setDevice(opt.gpuid)
    end
    cutorch.manualSeedAll(opt.manualSeed)

    if (opt.nEpochs == 0) then
        opt.nEpochs = math.huge
    end

    opt.optimState = {
        learningRate = opt.lr,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        dampening = 0,
        learningRateDecay = 1e-5,
        nesterov = true,
        beta1 = opt.beta1
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
