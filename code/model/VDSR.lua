require 'nn'
require 'cunn'

local function createModel(opt)
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization
    local seq = nn.Sequential
    local mul = nn.Mul
    local id = nn.Identity
    local join = nn.JoinTable
    local replicate = nn.Replicate
    local split = nn.SplitTable
    local cat = nn.ConcatTable
    local flat = nn.FlattenTable
    local cadd = nn.CAddTable

    local filt_recon = opt.filt_recon
    local pad_recon = (filt_recon-1)/2
    local nCh,nFeat = opt.nChannel,opt.nFeat

    local share = opt.vdsr_share_param
    local share_recon = opt.vdsr_share_recon


    local nlayerPerGroup = opt.nLayer / opt.vdsr_ngroup
    assert(nlayerPerGroup == math.floor(nlayerPerGroup),'nLayer is not divided by vdsr_ngroup')

    -- get_grou_indi returns group of layers used in usual sequential model, such as VDSR
    local function get_group_indi()
        local group_indi = seq()
        for i=1,nlayerPerGroup do
            group_indi:add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            group_indi:add(bnorm(nFeat))
            group_indi:add(relu(true))
        end
        return group_indi
    end
    -- group_shared is used in recursive layers that share parameters
    local group_shared = seq()
        :add(conv(nCh,nFeat, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))
        :add(relu(true))
    for i=1,nlayerPerGroup do
        group_shared:add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        group_shared:add(bnorm(nFeat))
        group_shared:add(relu(true))
    end
    group_shared:add(conv(nFeat,nCh, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))
    
    local function get_recon_indi()
        local recon_indi_1 = conv(nFeat,nCh, filt_recon,filt_recon, 1,1, pad_recon,pad_recon)
        local recon_indi_2 = conv(nCh,nCh, filt_recon,filt_recon, 1,1, pad_recon,pad_recon)

        if share then
            return recon_indi_2
        else
            return recon_indi_1
        end
    end
    local recon_shared_1 = conv(nFeat,nCh, filt_recon,filt_recon, 1,1, pad_recon,pad_recon)
    local recon_shared_2 = conv(nCh,nCh, filt_recon,filt_recon, 1,1, pad_recon,pad_recon)

    local function group() 
        if share then
            return group_shared
        else
            return get_group_indi()
        end
    end
    local function recon()
        if share_recon then
            if share then
                return recon_shared_2
            else
                return recon_shared_1
            end
        else
            return get_recon_indi()
        end
    end




    local model
    -- version 1: same as original paper
    if opt.vdsr_ver == 1 then
        model = seq()
            :add(conv(nCh,nFeat, filt_recon,filt_recon, 1,1, pad_recon, pad_recon))
            :add(relu(true))
        for i = 1,opt.nLayer do
            model:add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            model:add(bnorm(nFeat))
            model:add(relu(true))
        end
        model:add(conv(nFeat,nCh, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))

        model = seq()
            :add(cat()
                :add(model)
                :add(id())
            )
            :add(cadd())

    -- version 2: supervise every n'th intermediate feature maps individually with equal weight
    --            both sequential (e.g. VDSR) and recursive (e.g. DRCN) methods supported
    elseif opt.vdsr_ver == 2  then
        model = seq()
            :add(group())
            :add(recon())

        for i=1,opt.vdsr_ngroup-1 do
            model = seq()
                :add(group())
                :add(cat()
                    :add(model)
                    :add(recon())
                )
        end

        if not share then
            model:insert(conv(nCh,nFeat, 3,3, 1,1, 1,1),1)
            model:insert(relu(true),2)
        end

        if opt.vdsr_ngroup > 1 then
            model:add(flat())
            model:add(join(1))
        end

        model = seq()
            :add(cat()
                :add(model)
                :add(seq() -- skip connection
                    :add(replicate(opt.vdsr_ngroup))
                    :add(split(1))
                    :add(join(1))
                )
            )
            :add(cadd())
        
    -- version 3: Similar to ver 2, but supervise implicitly by doing weighted sum.
    --            Didn't group conv layers, since weights will find the optimal values through the backprops.
    --            If all the parameters are shared, the models becoms DRCN (J. Kim et al., CVPR 2016)
    elseif opt.vdsr_ver == 3  then
        model = seq()
            :add(group())
            :add(recon())
            :add(mul())

        for i=1,opt.vdsr_ngroup-1 do
            model = seq()
                :add(group())
                :add(cat()
                    :add(model)
                    :add(seq()
                        :add(recon())
                        :add(mul())
                    )
                )
        end

        if not share then
            model:insert(conv(nCh,nFeat, 3,3, 1,1, 1,1),1)
            model:insert(relu(true),2)
        end

        if opt.vdsr_ngroup > 1 then
            model:add(flat())
            model:add(cadd())
        end

        model = seq()
            :add(cat()
                :add(model)
                :add(id())
            )
            :add(cadd())
    end 

    return model
end

return createModel