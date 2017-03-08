clear
targetDir = 'img_target/latest';
scale = [2 3 4];

totalDir = dir(fullfile(targetDir));
for iSet = 1:length(totalDir)
    setName = totalDir(iSet).name;
    if (setName(1) == '.')
        continue;
    end
    setFull = fullfile(targetDir, setName);
    setDir = dir(setFull);
    for sc = 1:length(scale)
        mkdir(fullfile(setFull, ['X' num2str(scale(sc))])); 
    end
    for im = 1:length(setDir)
        imageName = setDir(im).name;
        imageDir = fullfile(setFull, imageName);
        if ((imageName(1) ~= '.') && (strcmp(imageName, 'Thumbs.db') == 0))
            original = imread(imageDir);
            for sc = 1:length(scale)
                sz = size(original);
                h = sz(1) - mod(sz(1), scale(sc));
                w = sz(2) - mod(sz(2), scale(sc));
                crop = original(1:h, 1:w, :);
                down = imresize(crop, 1 / scale(sc), 'bicubic');
                imwrite(down, fullfile(setFull, ['X' num2str(scale(sc))], imageName));
            end
        end
    end
end