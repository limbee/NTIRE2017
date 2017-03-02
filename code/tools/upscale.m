target = 'DIV2K_valid_LR_bicubic';
scale = [2 3 4];

for sc = 1:length(scale)
    inputDir = [target '/X' num2str(scale(sc))];
    targetDir = [target '/X' num2str(scale(sc)) 'b'];
    dirList = dir(inputDir);
    for fi = 3:length(dirList)
        imgName = fullfile(inputDir, dirList(fi).name);
        image = imread(imgName);
        resize = imresize(image, scale(sc));
        imwrite(resize, fullfile(targetDir, dirList(fi).name));
    end
end