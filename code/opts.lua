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
    cmd:option('-manualSeed',       0,                  'Manually set RNG seed')
    cmd:option('-nGPU',             1,                  'Number of GPUs to use by default')
    cmd:option('-gpuid',            1,                  'GPU id to use')
    cmd:option('-nThreads',         3,                  'Number of data loading threads')
    cmd:option('-save',             now,                'Subdirectory to save/log experiments in')
    cmd:option('-preTrained',       'nil',              'Directory of pre-trained model')
    -- Data
    cmd:option('-datadir',          '/var/tmp/dataset', 'Dataset location')
    cmd:option('-dataset',          'div2k',            'Dataset for training: div2k | imagenet')
    cmd:option('-datatype',         't7',               'Dataset type: png | t7 | t7pack')
    cmd:option('-dataSize',         'small',            'Input image size: small | big')
    cmd:option('-degrade',          'bicubic',          'Degrade type: bicubic | unknwon')
    cmd:option('-numVal',           10,                 'Number of images for validation')
    cmd:option('-rejection',        -1,                 'Enables patch rejection which has low gradient (uninformative)')
    cmd:option('-rejectionTarget',  'input',            'Reject patches based on input | target patch gradient')
    cmd:option('-colorAug',         'false',            'Apply color augmentation (brightness, contrast, saturation')
    cmd:option('-subMean',          'true',             'Data pre-processing: subtract mean')
    cmd:option('-mulImg',           255,                'Data pre-processing: multiply constant value to both input and output')
    -- Training
    cmd:option('-nEpochs',          300,                'Number of total epochs to run. 0: Infinite')
    cmd:option('-startEpoch',       0,                  'Manual epoch number for resuming the training. Default is the end')
    cmd:option('-lrDecay',          'step',             'Learning rate decaying method: step | exp | inv')
    cmd:option('-halfLife',         200e3,              'Half-life of learning rate: default is 200e3')
    cmd:option('-batchSize',        16,                 'Mini-batch size (1 = pure stochastic)')
    cmd:option('-patchSize',        96,                 'Training patch size')
    cmd:option('-scale',            '2',                'Super-resolution upscale factor')
    cmd:option('-valOnly',          'false',            'Run on validation set only')
    cmd:option('-trainOnly',        'false',            'Train without validation')
    cmd:option('-printEvery',       1e2,                'Print log every # iterations')
    cmd:option('-testEvery',        1e3,                'Test every # iterations')
    cmd:option('-load',             '.',                'Load saved training model, history, etc.')
    cmd:option('-clip',             -1,                 'Gradient clipping constant(theta)')
    cmd:option('-reset',            'false',            'Reset training')
    cmd:option('-chopShave',        10,                 'Shave width for chopForward')
    cmd:option('-chopSize',         16e4,               'Minimum chop size for chopForward')
    -- Optimization
    cmd:option('-optimMethod',      'ADAM',             'Optimization method')
    cmd:option('-lr',               1e-4,               'Initial learning rate')
    cmd:option('-lrLow',            1,                  'Relative learning rate of low frequency components')
    cmd:option('-lrHigh',           1,                  'Relative learning rate of high frequency components')
    cmd:option('-momentum',         0.9,                'SGD momentum')
    cmd:option('-beta1',            0.9,                'ADAM beta1')
    cmd:option('-beta2',            0.999,              'ADAM beta2')
    cmd:option('-epsilon',          1e-8,               'ADAM epsilon')
    cmd:option('-rho',              0.95,               'ADADELTA rho')
    -- Model
    cmd:option('-printModel',       'false',            'Print model at the start of the training')
    cmd:option('-netType',          'baseline',         'SR network architecture. Options: baseline | resnet | vdsr | msresnet')
    cmd:option('-filtsize',         3,                  'Filter size of convolutional layer')
    cmd:option('-nLayer',           20,                 'Number of convolution layer (for VDSR)')
    cmd:option('-nConv',            36,                 'Number of convolution layers excluding the beginning and end')
    cmd:option('-nResBlock',        16,                 'Number of residual blocks in SR network (for SRResNet, SRGAN)')
    cmd:option('-nChannel',         3,                  'Number of input image channels: 1 or 3')
    cmd:option('-nFeat',            64,                 'Number of feature maps in residual blocks in SR network')
    cmd:option('-upsample',         'espcnn',           'Upsampling method: deconv | espcnn')
    cmd:option('-trainNormLayer',   'false',            'Train normalization layer')
    cmd:option('-nOut',             1,                  'Number of output')
    cmd:option('-selOut',           -1,                 'Select output if there exists multiple outputs in model')
    cmd:option('-modelVer',         1,                  'Experimental model version')
    cmd:option('-act',              'relu',             'Activation function: relu | prelu | rrelu | elu | leakyrelu')
    cmd:option('-l',                1/8,                'Parameter l for RReLU')
    cmd:option('-u',                1/3,                'Parameter u for RReLU')
    cmd:option('-alpha',            1,                  'Parameter alpha for ELU')
    cmd:option('-negval',           1/100,              'Parameter negval for Leaky ReLU')
    cmd:option('-isSwap',           'false',            'Fast-swap for the models that generate multiple outputs')
    cmd:option('-mobranch',         1,                  'Position of the branch in MOResnet')
    cmd:option('-scaleRes',         1,                  'Scale each residuals in residual blocks')
    cmd:option('-nextnFeat',        256,                'ResNeXt nFeat')
    cmd:option('-nextC',            32,                 'ResNeXt cardinality')
    cmd:option('-nextF',            4,                  'ResNeXt branch features')
    -- Loss
    cmd:option('-abs',              1,                  'L1 loss weight')
    cmd:option('-smoothL1',         0,                  'Smooth L1 loss weight')
    cmd:option('-mse',              0,                  'MSE loss weight')
    cmd:option('-grad',             0,                  'Gradient loss weight')
    cmd:option('-grad2',            0,                  '2nd order Gradient loss weight')
    cmd:option('-gradDist',         'mse',              'Distance of gradient loss abs | mse')
    cmd:option('-gradPrior',        0,                  'Gradient prior weight')
    cmd:option('-gradPower',        0.8,                'Gradient prior power')
    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.printModel = opt.printModel == 'true'

    opt.colorAug = opt.colorAug == 'true'
    opt.subMean = opt.subMean == 'true'
    opt.divStd = opt.divStd == 'true'
    opt.trainNormLayer = opt.trainNormLayer == 'true'

    opt.valOnly = opt.valOnly == 'true'
    opt.trainOnly = opt.trainOnly == 'true'
    opt.reset = opt.reset == 'true'
    opt.isSwap = opt.isSwap == 'true'

    opt.scale = opt.scale:split('_')
    opt.psnrLabel = {}
    for i = 1, #opt.scale do
        opt.scale[i] = tonumber(opt.scale[i])
        table.insert(opt.psnrLabel, 'X' .. opt.scale[i])
    end

    if opt.load ~= '.' then 
        opt.save = opt.load
        if not paths.dirp(paths.concat('../experiment',opt.save)) then
            print(opt.load .. ' does not exist. Train new model.')
            opt.load = false
        end
    else
        opt.load = false
    end

    if opt.reset then
        assert(not opt.load, 'Cannot reset the training while loading a history')
        os.execute('rm -rf ../experiment/' .. opt.save)
    end

    opt.save = paths.concat('../experiment',opt.save)
    if not paths.dirp(opt.save) then
        paths.mkdir(opt.save)
        paths.mkdir(paths.concat(opt.save,'result'))
        paths.mkdir(paths.concat(opt.save,'model'))
    end

    torch.manualSeed(opt.manualSeed)
    torch.setdefaulttensortype('torch.FloatTensor')

    if opt.nGPU == 1 then
        cutorch.setDevice(opt.gpuid)
    end
    cutorch.manualSeedAll(opt.manualSeed)

    if opt.nEpochs == 0 then
        opt.nEpochs = math.huge
    end

    opt.optimState = {
        learningRate = opt.lr,
        momentum = opt.momentum,
        dampening = 0,
        nesterov = true,
        beta1 = opt.beta1,
        beta2 = opt.beta2,
        epsilon = opt.epsilon,
        rho = opt.rho
    }

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
