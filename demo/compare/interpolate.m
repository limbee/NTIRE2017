clear;

sets = {'Set5', 'Set14', 'Urban100', 'B100', 'val'};
apath = '.';

for s = sets
    set = s{1,1};
    if ~exist(fullfile(apath, 'Bicubic', set), 'dir')
        mkdir(fullfile(apath, 'Bicubic', set));
    end
    images = dir(fullfile(apath, 'GT', set, '*.png'));
    for img = images'
		imgName = img.name;
		gt = imread(fullfile(apath, 'GT', set, imgName));
        for sc = [2 3 4]
            if ~exist(fullfile(apath, 'Bicubic', set, strcat('X', num2str(sc))), 'dir')
                mkdir(fullfile(apath, 'Bicubic', set, strcat('X', num2str(sc))));
            end
            [h, w, ch] = size(gt);
            sh = floor(h / sc) * sc;
            sw = floor(w / sc) * sc;
            gt = gt(1:sh, 1:sw, :);
            
            lr = imresize(gt, 1/sc, 'bicubic');
            ilr = imresize(lr, sc, 'bicubic');
            saveName = fullfile(apath, 'Bicubic', set, strcat('X', num2str(sc)), imgName);
            imwrite(ilr, saveName);
        end
    end
end
