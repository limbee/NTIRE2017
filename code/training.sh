# Bicubic scale 2
#th main.lua -scale 2 -nFeat 256 -nResBlcok 36 -patchSize 96 -scaleRes 0.1 -skipBatch 3

# Bicubic scale 3 from pre-trained bicubic scale 2 model
#th main.lua -scale 3 -netType resnet_cu -nFeat 256 -nResBlock 36 -patchSize 144 -scaleRes 0.1 -skipBatch 3 -preTrained ../demo/model/bicubic_x2.t7

# Bicubic scale 4 from pre-trained bicubic scale 2 model
#th main.lua -scale 4 -netType resnet_cu -nFeat 256 -nResBlock 36 -patchSize 192 -scaleRes 0.1 -skipBatch 3 -preTrained ../demo/model/bicubic_x2.t7



# Unknown scale 2 with L1 loss from pre-tranined bicubic scale 2 model
#th main.lua -scale 2 -degrade unknown -nFeat 256 -nResBlcok 36 -patchSize 96 -scaleRes 0.1 -skipBatch 3 # -preTrained ../demo/model/bicubic_x2.t7

# Unknown scale 2 with G1 loss from pre-tranined bicubic scale 2 model
#th main.lua -scale 2 -degrade unknown -grad 1 -gradDist abs -nFeat 256 -nResBlcok 36 -patchSize 96 -scaleRes 0.1 -skipBatch 3 # -preTrained ../demo/model/bicubic_x2.t7



# Unknown scale 3 with L1 loss from pre-tranined bicubic scale 2 model
#th main.lua -scale 3 -netType resnet_cu -degrade unknown -nFeat 256 -nResBlcok 36 -patchSize 144 -scaleRes 0.1 -skipBatch 3 -preTrained ../demo/model/bicubic_x2.t7

# Unknown scale 3 with G1 loss from pre-tranined bicubic scale 2 model
#th main.lua -scale 3 -netType resnet_cu -degrade unknown -grad 1 -gradDist abs -nFeat 256 -nResBlcok 36 -patchSize 144 -scaleRes 0.1 -skipBatch 3 -preTrained ../demo/model/bicubic_x2.t7



# Unknown scale 4 with L1 loss from pre-tranined bicubic scale 2 model
#th main.lua -scale 4 -netType resnet_cu -degrade unknown -nFeat 256 -nResBlcok 36 -patchSize 192 -scaleRes 0.1 -skipBatch 3 -preTrained ../demo/model/bicubic_x2.t7

# Unknown scale 4 with G1 loss from pre-tranined bicubic scale 2 model
#th main.lua -scale 4 -netType resnet_cu -degrade unknown -grad 1 -gradDist abs -nFeat 256 -nResBlcok 36 -patchSize 192 -scaleRes 0.1 -skipBatch 3 -preTrained ../demo/model/bicubic_x2.t7



# Bicubic multicale
#th main.lua -scale 2_3_4 -netType multiscale -nResBlock 80 -patchSize 64 -multiPatch true -skipBatch 3

# Unknown multiscale with L1 loss from pre-trained bicubic multiscale model
#th main.lua -scale 2_3_4 -degrade unknown -netType multiscale_unknown -nResBlock 80 -patchSize 64 -multiPatch true -skipBatch 3 -preTrained ../demo/model/bicubic_multiscale.t7

# Unknown multiscale with L1 loss from pre-trained bicubic multiscale model
#th main.lua -scale 2_3_4 -degrade unknown -netType multiscale_unknown -grad 1 -gradDist abs -nResBlock 80 -patchSize 64 -multiPatch true -skipBatch 3 -preTrained ../demo/model/bicubic_multiscale.t7