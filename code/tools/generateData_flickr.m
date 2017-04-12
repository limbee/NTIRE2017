scale = [2 3 4];
degrade = 'bicubic';
apath = '/dataset/Flickr2K';
hrDir = fullfile(apath, 'Flickr2K_HR');
lrDir = fullfile(apath, strcat('Flickr2K_LR_', degrade));

for sc = 1:length(scale)
    lrSubDir = fullfile(lrDir, strcat('X', num2str(scale(sc))));
    if ~exist(lrSubDir, 'dir')
        mkdir(lrSubDir);
    end
end

HRs = dir(fullfile(hrDir, '*.png'));
for img = 1:length(HRs)
    imgName = HRs(img).name;
    imgFull = fullfile(hrDir, imgName);
    hrImg = imread(imgFull);

    [h, w, c] = size(hrImg);
    ch = floor(h / 12) * 12;
    cw = floor(w / 12) * 12;
    hrImg = hrImg(1:ch, 1:cw, :);
    imwrite(hrImg, imgFull);

    for sc = 1:length(scale)
        lrImg = zeros(0);
        if degrade == 'bicubic'
            [h, w, c] = size(hrImg);
            ch = floor(h / scale(sc)) * scale(sc);
            cw = floor(w / scale(sc)) * scale(sc);
            cropped = hrImg(1:ch, 1:cw, :);
            lrImg = imresize(cropped, 1/scale(sc), 'bicubic');
        end
        strsc = num2str(scale(sc));
        [ps, imgNamewoExt, ext] = fileparts(imgName);
        imgSave = strcat(imgNamewoExt, 'x', strsc, ext);
        imwrite(lrImg, fullfile(lrDir, strcat('X', strsc), imgSave));
    end
end

exit()