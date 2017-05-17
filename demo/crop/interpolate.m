clear;

for degrade = {'bicubic', 'unknown'}
    for sc = [2 3 4]
        for i = 791:800
            name = strcat('/var/tmp/dataset/DIV2K/DIV2K_train_LR_', degrade{1,1}, '/X', num2str(sc), '/0', num2str(i), 'x', num2str(sc), '.png');
            lr = imread(name);
            bic = imresize(lr, sc, 'bicubic');
            saveName = strcat('interpolate_', degrade{1,1}, '_0', num2str(i), 'x', num2str(sc), '.png');
            imwrite(bic, saveName);
        end
    end
end