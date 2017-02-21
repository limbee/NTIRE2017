model = 'SRResNet';
clearvars -except model

resDir = fullfile('img_output',model);
tarDir = fullfile('img_target',model);

dirList = dir(resDir);
dirList = dirList(~ismember({dirList.name},{'.','..'}));


for iDir = 1:length(dirList)
    dirName = dirList(iDir).name;
    disp(dirName);
    imgList = dir(fullfile(resDir,dirName));
    imgList = imgList(~ismember({imgList.name},{'.','..'}));
    PSNR = 0;
    SSIM = 0;

    for iImg = 1:length(imgList)
        fileName = imgList(iImg).name;
        fprintf('[%d/%d][%d/%d] %s\n', iDir,length(dirList), iImg,length(imgList), fileName);

        resImg = rgb2ycbcr(imread(fullfile(resDir,dirName,fileName)));
        tarImg = rgb2ycbcr(imread(fullfile(tarDir,dirName,fileName)));
        resImg = squeeze(resImg(:,:,1));
        tarImg = squeeze(tarImg(:,:,1));

        PSNR = PSNR + psnr(resImg,tarImg);
        SSIM = SSIM + ssim(resImg,tarImg);
    end
    fprintf('PSNR: %.2f, SSIM: %.4f\n\n', PSNR/length(imgList), SSIM/length(imgList));
end