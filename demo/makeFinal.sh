# [EDSR+]

# Bicubic scale 2
# th test.lua -type val -dataset DIV2K -model EDSR_x2 -scale 2 -selfEnsemble true

# Bicubic scale 3
# th test.lua -type val -dataset DIV2K -model EDSR_x3 -scale 3 -selfEnsemble true

# Bicubic scale 4
# th test.lua -type val -dataset DIV2K -model EDSR_x4 -scale 4 -selfEnsemble true



# [MDSR+]

# Bicubic multiscale (Note that scale 2, 3, 4 share the same model!)

# For scale 2
# th test.lua -type val -dataset DIV2K -model MDSR -scale 2 -selfEnsemble true

# For scale 3
# th test.lua -type val -dataset DIV2K -model MDSR -scale 3 -selfEnsemble true

# For scale 4
# th test.lua -type val -dataset DIV2K -model MDSR -scale 4 -selfEnsemble true
