clear;

%You can select the scale from here
scale = [1.5 2 3 4];

%You can select the degrading operator from here
degrade = 'bicubic';

%Folder where 'DIV2K' folder exists
%apath = '/var/tmp/dataset';
apath = '../../../';
hrDir = fullfile(apath, 'DIV2K', 'DIV2K_train_HR');

ff = fullfile(apath, 'DIV2K', strcat('DIV2K_train_LR_', degrade));
lrDir = ff;
for sc = 1:length(scale)
    folderName = fullfile(lrDir, strcat('X', num2str(scale(sc))));
    if (exist(folderName, 'dir') ~= 7)
        mkdir(folderName);
    end
end

gtDir = dir(fullfile(hrDir, '*.png'));
for img = 1:length(gtDir)
    imgName = gtDir(img).name;
    imgFull = fullfile(hrDir, imgName);
    hrImg = imread(imgFull);
    for sc = 1:length(scale)
        lrImg = zeros(0);
        %You can define your own degrading operator here
        if (degrade == 'bicubic')
            [h, w, c] = size(hrImg);
            ch = floor(h / scale(sc)) * scale(sc);
            cw = floor(w / scale(sc)) * scale(sc);
            cropped = hrImg(1:ch, 1:cw, :);
            lrImg = imresize(cropped, 1 / scale(sc), 'bicubic');
        end
        strsc = num2str(scale(sc));
        [ps, imgNamewoExt, ext] = fileparts(imgName);
        imgSave = strcat(imgNamewoExt, 'x', strsc, ext);
        ff = fullfile(lrDir, strcat('X', strsc), imgSave);
        imwrite(lrImg, ff);
    end
end

