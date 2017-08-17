clear;
close all;

tag = {'plants', 'nature'};
setName = '../../flickr';
fileList = dir(fullfile(setName, '*.png'));

fig = figure;
for idx = 1:length(fileList);
    fileName = fileList(idx).name;
    
    isOkay = false;
    for t = 1:length(tag)
        if isempty(strfind(fileName, tag(t))) == 0
            isOkay = true;
            break;
        end
    end
    if isOkay == false
        continue;
    end
    
    fileFull = fullfile(setName, fileName);
    disp(fileName);
    img = imresize(imread(fileFull), 0.75, 'bicubic');
    imshow(img);
    fig.Name = fileFull;
    if getKey(fig) == 'x'
        movefile(fileFull, '../../garbage');
    else
        movefile(fileFull, '../../fine');
    end
end