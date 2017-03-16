apath = '/var/tmp/dataset/DIV2K';

HRpath = 'DIV2K_train_HR';
LRpath = {'DIV2K_train_LR_bicubic', 'DIV2K_train_LR_unknown'};
scale = {'X2', 'X3', 'X4'};

dirList = dir(fullfile(apath, HRpath));
for hr = 1:length(dirList)
    imgName = dirList(hr).name;
    if (length(strfind(imgName, '.png')) ~= 0)
        img = imread(fullfile(apath, HRpath, imgName));
        rot = imrotate(img, -45, 'bicubic', 'crop');
        name = fullfile(apath, HRpath, [imgName(1:(end - 4)) 'r.png']);
        imwrite(rot, name);
        disp(name);
    end
end

for lrs = 1:length(LRpath)
    for sc = 1:length(scale)
        lp = char(LRpath(lrs));
        sp = char(scale(sc));
        dirList = dir(fullfile(apath, lp, sp));
        for d = 1:length(dirList)
            imgName = dirList(d).name;
            if (length(strfind(imgName, '.png')) ~= 0)
                img = imread(fullfile(apath, lp, sp, imgName));
                rot = imrotate(img, -45, 'bicubic', 'crop');
                name = fullfile(apath, lp, sp, [imgName(1:(end - 4)) 'r.png']);
                imwrite(rot, name);
                disp(name);
            end
        end
    end
end
