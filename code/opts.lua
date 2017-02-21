require 'cutorch'

local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()

    cmd:text()
    cmd:text('Super resolution using perceptual loss, GAN, and residual network architecture')
    cmd:text('This is an implementation of the paper: Photo-Realistic Image Super-Resolution Using a Generative Adversarial Network (C. Ledig, 2016)')
    cmd:text()
    cmd:text('Options:')
    -- Global
    cmd:option('-manualSeed', 0,          'Manually set RNG seed')
    cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
    cmd:option('-gpuid',      2,            'GPU id to use')
    cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
    cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
    cmd:option('-gen',        'gen',      'Path to save generated files')
    cmd:option('-nThreads',    7,         'number of data loading threads')
	cmd:option('-save',       os.date("%Y-%m-%d_%H-%M-%S"),       'subdirectory to save/log experiments in')
    cmd:option('-defaultType', 'torch.FloatTensor', 'Default data type')
    -- Data
	cmd:option('-dataset', 'imagenet', 'dataset for training: imagenet | coco | 91 | 291')
    cmd:option('-trainset', 'val', 'train set(500k train, 50k val): train | val')
    cmd:option('-valset', 'Set5', 'validation set: val | Set5 | Set14 | B100 | Urban100')
    cmd:option('-sigma',    3,      'Sigma used for gaussian blur before shrinking an image.')
    cmd:option('-inter',    'bicubic', 'Interpolation method used for downsizing an image: bicubic | inter_area')
    cmd:option('-matlab',   'true',     'Use input image downsized by matlab function imresize() with bicubic interpolation')
    -- Training
    cmd:option('-nEpochs',       0,       'Number of total epochs to run')
    cmd:option('-epochNumber',   1,       'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',     16,      'mini-batch size (1 = pure stochastic)')
    cmd:option('-patchSize',    96,         'Training patch size')
    cmd:option('-scale',        4,          'Super-resolution upscale factor')
    cmd:option('-testOnly',    'false', 'Run on validation set only')
    cmd:option('-printEvery',   1e2,       'Print log every # iterations')
    cmd:option('-testEvery',    1e3,       'Test every # iterations')
    cmd:option('-load',         '.',     'Load saved training model, history, etc.')
    cmd:option('-clip',         -1,    'Gradient clipping constant(theta)')
    -- Optimization
    cmd:option('-optimMethod',  'ADAM',  'Optimization method')
    cmd:option('-lr',         1e-4, 'initial learning rate')
	cmd:option('-momentum', 0.9, 'momentum (SGD only)')
	cmd:option('-beta1', 0.9, 'ADAM momentum')
    cmd:option('-weightDecay', 0, 'weight decay')
    cmd:option('-optimMethod_d',  'ADAM',  'Optimization method')
    cmd:option('-lr_d',     1e-4,   'initial learning rate for discriminator')
    cmd:option('-momentum_d',0.9, 'momentum for discriminator')
    cmd:option('-beta1_d',  0.9, 'ADAM momentum for discriminator')
    cmd:option('-weightDecay_d', 0, 'weight decay for discriminator')
    -- Model
    cmd:option('-netType',      'ResNet', 'Generator network architecture. Options: ResNet | VDSR')
    cmd:option('-pre_act',      'false',    'Pre-activation architecture (for ResNet)')
    cmd:option('-bottleneck',   'false',  'Use bottleneck architecture (for ResNet and preResNet)')
    cmd:option('-nLayer',       20,       'Number of convolution layer (for VDSR)')
    cmd:option('-nResBlock',    16,     'Number of residual blocks in generator network (for SRResNet, SRGAN)')
    cmd:option('-nChannel',     3,      'Number of input image channels: 1 or 3')
    cmd:option('-nFeat',    64,     'Number of feature maps in residual blocks in generator network')
    cmd:option('-normalize',   'false',   'Normalize pixel values to be zero mean, unit std')
    cmd:option('-upsample',  'shuffle',   'Upsampling method: full | bilinear | shuffle')
    cmd:option('-filt_deconv',  4,      'filter size for deconvolution layer')
    cmd:option('-filt_recon',  3,      'filter size for reconstruction layer')
    cmd:option('-vdsr_ver',     1,      'Version of experimental VDSR model')
    cmd:option('-vdsr_ngroup',  1,      'Number of convolution groups (for VDSR version 2,3)')
    cmd:option('-vdsr_share_param',  'false',      'Share parameters between group of layers. Useful when making DRCN model')
    cmd:option('-vdsr_share_recon',  'false',      'Share parameters between reconstruction layers')
    -- Loss
    cmd:option('-abs',  0,  'L1 loss weight')
    cmd:option('-smoothL1', 0, 'Smooth L1 loss weight')
    cmd:option('-mse',   1,  'MSE loss weight')
    cmd:option('-perc',   0,  'VGG loss weight (perceptual loss)')
    cmd:option('-adv',   0,  'Adversarial loss weight: 1e-3 in the paper')
    cmd:option('-tv',   0,  'Total variation regularization loss weight: 2e-8 in the paper')
        -- VGG loss
    cmd:option('-vggDepth', '5-4', 'Depth of pre-trained VGG for use in perceptual loss')
        -- Adversarial loss
    cmd:option('-negval',       0.2,    'Negative value parameter for Leaky ReLU in discriminator network')
    cmd:option('-filtsizeD',    3,    'Filter size of stride 2 convolutions in discriminator network')

    ---------- Model options ----------------------------------
    cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
    cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.pre_act = opt.pre_act=='true'
    opt.sharedGradInput = opt.sharedGradInput=='true'
    opt.optnet = opt.optnet=='true'
    opt.bottleneck = opt.bottleneck=='true'
    opt.normalize = opt.normalize=='true'
    opt.matlab = opt.matlab=='true'
    opt.vdsr_share_param = opt.vdsr_share_param=='true'
    opt.vdsr_share_recon = opt.vdsr_share_recon=='true'

    if type(opt.dataset)=='number' then opt.dataset = tostring(opt.dataset) end

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

    torch.setdefaulttensortype('torch.FloatTensor')
    --torch.setnumthreads(1)
    torch.manualSeed(opt.manualSeed)

    if opt.nGPU == 1 then
        cutorch.setDevice(opt.gpuid)
    end
    cutorch.manualSeedAll(opt.manualSeed)

    if opt.nEpochs < 1 then opt.nEpochs = math.huge end

    torch.setdefaulttensortype(opt.defaultType)
    if opt.nGPU > 0 then
        opt.operateType = 'torch.CudaTensor'
    else
        opt.operateType = opt.defaultType
    end

    opt.optimState_G = {
        learningRate = opt.lr,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        dampening = 0,
        learningRateDecay = 1e-5,
        nesterov = true,
        beta1 = opt.beta1
    }
	if opt.optimMethod == 'SGD' then opt.optimState_G.method = optim.sgd
    elseif opt.optimMethod == 'ADADELTA' then opt.optimState_G.method = optim.adadelta
	elseif opt.optimMethod == 'ADAM' then opt.optimState_G.method = optim.adam
	elseif opt.optimMethod == 'RMSPROP' then opt.optimState_G.method = optim.rmsprop
	else error('unknown optimization method') end  

    if opt.adv > 0 then
        opt.optimState_D = {
            learningRate = opt.lr_d,
            weightDecay = opt.weightDecay_d,
            momentum = opt.momentum_d,
            dampening = 0,
            learningRateDecay = 1e-5,
            nesterov = true,
            beta1 = opt.beta1_d
        }
        if opt.optimMethod_d == 'SGD' then opt.optimState_D.method = optim.sgd
        elseif opt.optimMethod_d == 'ADADELTA' then opt.optimState_D.method = optim.adadelta
        elseif opt.optimMethod_d == 'ADAM' then opt.optimState_D.method = optim.adam
        elseif opt.optimMethod_d == 'RMSPROP' then opt.optimState_D.method = optim.rmsprop
        else error('unknown optimization method') end  
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