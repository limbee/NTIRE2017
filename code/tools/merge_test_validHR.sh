tar -xzf valid.tar.gz
mv valid/DIV2K_valid_HR/*.png DIV2K_train_HR/
mv valid/DIV2K_test_* ./
mv DIV2K_valid_LR_bicubic/X2/* DIV2K_train_LR_bicubic/X2/
mv DIV2K_valid_LR_bicubic/X3/* DIV2K_train_LR_bicubic/X3/
mv DIV2K_valid_LR_bicubic/X4/* DIV2K_train_LR_bicubic/X4/
mv DIV2K_valid_LR_unknown/X2/* DIV2K_train_LR_unknown/X2/
mv DIV2K_valid_LR_unknown/X3/* DIV2K_train_LR_unknown/X3/
mv DIV2K_valid_LR_unknown/X4/* DIV2K_train_LR_unknown/X4/
rm -rf DIV2K_valid_LR_*
rm -rf valid
rm valid.tar.gz
sudo chmod -R 777 ./
