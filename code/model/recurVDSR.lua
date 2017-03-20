require 'nn'
require 'cunn'

local function createModel(opt)
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization

    local filt_recon = 3
    local pad_recon = (filt_recon-1)/2


    local model = nn.Sequential()
    local model0 = nn.ConcatTable()
    local model1 = nn.Sequential()
    local model2 = nn.Sequential()
    local model3 = nn.Sequential()
    local model4 = nn.Sequential()
    local model5 = nn.Sequential()
    local main = nn.Sequential()


    local conv1 = nn.Sequential()
  
    conv1:add(conv(opt.nChannel,opt.nFeat, filt_recon,filt_recon, 1,1, pad_recon, pad_recon))
    conv1:add(relu(true))
    for i = 1,opt.nLayer do
        conv1:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))
        conv1:add(bnorm(opt.nFeat))
        conv1:add(relu(true))
    end
    conv1:add(conv(opt.nFeat,opt.nChannel, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))

    --local conv2 = conv1:clone()
    --local conv3 = conv1:clone()
    --local conv4 = conv1:clone()
    --local conv5 = conv1:clone()
    local conv2 = nn.Sequential()
  
    conv2:add(conv(opt.nChannel,opt.nFeat, filt_recon,filt_recon, 1,1, pad_recon, pad_recon))
    conv2:add(relu(true))
    for i = 1,opt.nLayer do
        conv2:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))
        conv2:add(bnorm(opt.nFeat))
        conv2:add(relu(true))
    end
    conv2:add(conv(opt.nFeat,opt.nChannel, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))
--[[

    local conv3 = nn.Sequential()
  
    conv3:add(conv(opt.nChannel,opt.nFeat, filt_recon,filt_recon, 1,1, pad_recon, pad_recon))
    conv3:add(relu(true))
    for i = 1,opt.nLayer/2 do
        conv3:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))
        conv3:add(bnorm(opt.nFeat))
        conv3:add(relu(true))
    end
    conv3:add(conv(opt.nFeat,opt.nChannel, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))

    local conv4 = nn.Sequential()
  
    conv4:add(conv(opt.nChannel,opt.nFeat, filt_recon,filt_recon, 1,1, pad_recon, pad_recon))
    conv4:add(relu(true))
    for i = 1,opt.nLayer/2 do
        conv4:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))
        conv4:add(bnorm(opt.nFeat))
        conv4:add(relu(true))
    end
    conv4:add(conv(opt.nFeat,opt.nChannel, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))

]]--

    local concat = nn.ConcatTable()
    concat:add(nn.Identity()):add(conv1)

 	model1:add(nn.Sequential()
			:add(concat)
			:add(nn.CAddTable())
		)

	model2:add(nn.Sequential()
		 	 :add(nn.ConcatTable()
		 		:add(nn.Identity()):clone()
				:add(nn.Sequential()
					:add(concat)
					:add(nn.CAddTable())
					:add(conv2)
					)
			  )
		 	:add(nn.CAddTable())
		)
		
			--[[

	model3:add(nn.Sequential()
			 :add(nn.ConcatTable()
				 :add(nn.Identity()):clone()
			 	 :add(nn.Sequential()
				 	 :add(nn.ConcatTable()
	 					:add(nn.Identity()):clone()
						:add(nn.Sequential()
							:add(concat)
							:add(nn.CAddTable())
							:add(conv2)
							)
					  )
				:add(nn.CAddTable())
				:add(conv3)
				)
		 	)
		:add(nn.CAddTable())
		)

	model4:add(nn.Sequential()
	 		  :add(nn.ConcatTable()
			 		:add(nn.Identity()):clone()
					:add(nn.Sequential()
						 :add(nn.ConcatTable()
		 					 :add(nn.Identity()):clone()
						 	 :add(nn.Sequential()
							 	 :add(nn.ConcatTable()
	 			 					:add(nn.Identity()):clone()
			 						:add(nn.Sequential()
										:add(concat)
										:add(nn.CAddTable())
										:add(conv2)
										)
								  )
							 	:add(nn.CAddTable())
								:add(conv3)
								)
						 	)
						:add(nn.CAddTable())
						:add(conv4)
						)
				 	)
				:add(nn.CAddTable())
			)



	model5:add(nn.Sequential()
			:add(nn.ConcatTable()
			 	:add(skip):clone()
			 	:add(nn.Sequential()
			 		  :add(nn.ConcatTable()
	 			 		:add(skip):clone()
						:add(nn.Sequential()
							 :add(nn.ConcatTable()
		 						 :add(skip):clone()
				 			 	 :add(nn.Sequential()
								 	 :add(nn.ConcatTable()
	 				 					:add(skip):clone()
			 							:add(nn.Sequential()
											:add(concat)
											:add(nn.CAddTable())
											:add(conv2)
											)
									  )
								 	:add(nn.CAddTable())
									:add(conv3)
									)
							 	)
							:add(nn.CAddTable())
							:add(conv4)
							)
					 	)
					:add(nn.CAddTable())
					:add(conv5)
				 	)
				)
			:add(nn.CAddTable())
			)
]]--


    model0:add(model1)
    model0:add(model2)
    --model0:add(model3)
    --model0:add(model4)
    --model0:add(model5)

    model:add(model0)
    model:add(nn.FlattenTable())
--print(model)
    return model
end

return createModel