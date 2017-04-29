# Bicubic scale 2
#th test.lua -type test -model bicubic_x2.t7 -scale 2 -selfEnsemble true -type val

# Bicubic scale 3
th test.lua -type test -model bicubic_x3.t7 -scale 3 -selfEnsemble true -type val

# Bicubic scale 4
th test.lua -type test -model bicubic_x4.t7 -scale 4 -selfEnsemble true -type val



# Unknown scale 2
th test.lua -type test -model unknown_x2_1.t7+unknown_x2_2.t7 -scale 2 -degrade unknown -type val

# Unknown scale 3
th test.lua -type test -model unknown_x3_1.t7+unknown_x3_2.t7 -scale 3 -degrade unknown -type val

# Unknown scale 4
th test.lua -type test -model unknown_x4_1.t7+unknown_x4_2.t7 -scale 4 -degrade unknown -type val



# Bicubic multiscale (Note that scale 2, 3, 4 share the same model!)

# For scale 2
#th test.lua -type test -model bicubic_multiscale -scale 2 -selfEnsemble true -type val

# For scale 3
th test.lua -type test -model bicubic_multiscale -scale 3 -selfEnsemble true -type val

# For scale 4
th test.lua -type test -model bicubic_multiscale -scale 4 -selfEnsemble true -type val



# Unknown multiscale (Note that scale 2, 3, 4 share the same model!)

# For scale 2
th test.lua -type test -model unknown_multiscale_1.t7+unknown_multiscale_2.t7 -scale 2 -degrade unknown -type val

# For scale 3
th test.lua -type test -model unknown_multiscale_1.t7+unknown_multiscale_2.t7 -scale 3 -degrade unknown -type val

# For scale 4
th test.lua -type test -model unknown_multiscale_1.t7+unknown_multiscale_2.t7 -scale 4 -degrade unknown -type val