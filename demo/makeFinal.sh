#Before starting, please place these files in the demo/model folder
#Bx2_L1.t7
#Bx2_G1.t7
#Bx3_L1.t7
#Bx3_L1.t7
#Bx4_G1.t7
#Bx4_G1.t7
#Ux2_L1.t7
#Ux2_G1.t7
#Ux3_L1.t7
#Ux3_L1.t7
#Ux4_G1.t7
#Ux4_G1.t7
#multiscale.t7

#For bicubic x2
#th test.lua -type test -model Bx2_L1.t7+Bx2_G1.t7 -degrade bicubic -scale 2 -nGPU 2 -selfEnsemble true -chopShave 10 -chopSize 16e4

#For bicubic x3
#th test.lua -type test -model Bx3_L1.t7+Bx3_G1.t7 -degrade bicubic -scale 3 -nGPU 2 -selfEnsemble true -chopShave 10 -chopSize 16e4

#For bicubic x4
#th test.lua -type test -model Bx4_L1.t7+Bx4_G1.t7 -degrade bicubic -scale 4 -nGPU 2 -selfEnsemble true -chopShave 10 -chopSize 16e4

#For unknown x2
#th test.lua -type test -model Ux2_L1.t7+Ux2_G1.t7 -degrade unknown -scale 2 -nGPU 2 -selfEnsemble false -chopShave 10 -chopSize 16e4

#For unknown x3
#th test.lua -type test -model Ux3_L1.t7+Ux3_G1.t7 -degrade unknown -scale 3 -nGPU 2 -selfEnsemble false -chopShave 10 -chopSize 16e4

#For unknown x4
#th test.lua -type test -model Ux4_L1.t7+Ux4_G1.t7 -degrade unknown -scale 4 -nGPU 2 -selfEnsemble false -chopShave 10 -chopSize 16e4

#For multiscale
#th test.lua -type test -model multiscale.t7 -degrade bicubic -scale 2 -swap 1 -nGPU 2 -selfEnsemble true -chopShave 10 -chopSize 16e4
#th test.lua -type test -model multiscale.t7 -degrade bicubic -scale 3 -swap 2 -nGPU 2 -selfEnsemble true -chopShave 10 -chopSize 16e4
#th test.lua -type test -model multiscale.t7 -degrade bicubic -scale 4 -swap 3 -nGPU 2 -selfEnsemble true -chopShave 10 -chopSize 16e4
