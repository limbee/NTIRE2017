th test.lua -type val -model SRresnet -scale 2 -feature true
#th test.lua -type val -model Baseline -scale 2 -feature true
#paper SRresnet
#th test.lua -type val -model 1 -scale 4
#th test.lua -type bench -model 1 -scale 4
#th test.lua -type val -model 2 -scale 4
#th test.lua -type bench -model 2 -scale 4

#imagenet
#th test.lua -type bench -model imagenet -scale 4 
#th test.lua -type bench -model model_bicubic_imagenetx2 -scale 2
#th test.lua -type test -degrade unknown -model model_unknown_DIV2Kx3 -scale 3
#th test.lua -type test -degrade unknown -model model_unknown_DIV2Kx4 -scale 4

# Bicubic scale 2
#th test.lua -type test -model bicubic_x2.t7 -scale 2 -selfEnsemble true

# Bicubic scale 3
#th test.lua -type test -model bicubic_x3.t7 -scale 3 -selfEnsemble true

# Bicubic scale 4
#th test.lua -type test -model bicubic_x4.t7 -scale 4 -selfEnsemble true



# Unknown scale 2
#th test.lua -type test -model unknown_x2_1.t7+unknown_x2_2.t7 -scale 2 -degrade unknown

# Unknown scale 3
#th test.lua -type test -model unknown_x3_1.t7+unknown_x3_2.t7 -scale 3 -degrade unknown

# Unknown scale 4
#th test.lua -type test -model unknown_x4_1.t7+unknown_x4_2.t7 -scale 4 -degrade unknown



# Bicubic multiscale (Note that scale 2, 3, 4 share the same model!)

# For scale 2
#th test.lua -type test -model bicubic_multiscale -scale 2 -selfEnsemble true

# For scale 3
#th test.lua -type test -model bicubic_multiscale -scale 3 -selfEnsemble true

# For scale 4
#th test.lua -type test -model bicubic_multiscale -scale 4 -selfEnsemble true



# Unknown multiscale (Note that scale 2, 3, 4 share the same model!)

# For scale 2
#th test.lua -type test -model unknown_multiscale_1.t7+unknown_multiscale_2.t7 -scale 2 -degrade unknown

# For scale 3
#th test.lua -type test -model unknown_multiscale_1.t7+unknown_multiscale_2.t7 -scale 3 -degrade unknown

# For scale 4
#th test.lua -type test -model unknown_multiscale_1.t7+unknown_multiscale_2.t7 -scale 4 -degrade unknown
