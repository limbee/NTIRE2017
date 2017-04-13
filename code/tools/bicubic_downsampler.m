scale = [2 3 4];
degrade = 'bicubic';
%dataset = 'DIV2K';
%apath = '/var/tmp';
dataset = 'Flickr2K';
apath = '/dataset';
hrDir = fullfile(apath, dataset, strcat(dataset, '_HR'));
lrDir = fullfile(apath, dataset, strcat(dataset, '_LR_', degrade));

for sc = 1:length(scale)
    lrSubDir = fullfile(lrDir, strcat('X', num2str(scale(sc))));
    if ~exist(lrSubDir, 'dir')
        mkdir(lrSubDir);
    end
end

hrImgs = dir(fullfile(hrDir, '*.png'));
for idxImg = 1:length(hrImgs)
    if mod(idxImg, 10) == 0
        fprintf('Processed %d / %d images\n', idxImg, length(hrImgs));
    end
    imgName = hrImgs(idxImg).name;
    hrImg = imread(fullfile(hrDir, imgName));

    if strcmp(dataset, 'Flickr2K')
        [h, w, ~] = size(hrImg);
        ch = floor(h / 12) * 12;
        cw = floor(w / 12) * 12;
        hrImg = hrImg(1:ch, 1:cw, :);
        imwrite(hrImg, fullfile(hrDir, imgName));
    end

    for sc = 1:length(scale)
        if strcmp(degrade, 'bicubic')
            [h, w, ~] = size(hrImg);
            ch = floor(h / scale(sc)) * scale(sc);
            cw = floor(w / scale(sc)) * scale(sc);
            cropped = hrImg(1:ch, 1:cw, :);
            lrImg = imresize(cropped, 1/scale(sc), 'bicubic');
        end
        strsc = num2str(scale(sc));
        [~, imgNamewoExt, ext] = fileparts(imgName);
        imgSave = strcat(imgNamewoExt, 'x', strsc, ext);
        imwrite(lrImg, fullfile(lrDir, strcat('X', strsc), imgSave));
    end
end

exit()