% Calculates PSNR and SSIM of each SR outputs
% on y channel, border crop = (scale) pixels

clear; clc;

apath = '/SR_compare';
works = {'Bicubic', 'Aplus', 'SRCNN', 'VDSR', 'SRResNet_reproduce', 'Ours_Single', 'Ours_Multi'};
testSets = {'Set5', 'Set14', 'B100', 'Urban100', 'val'};
[a, len] = size(testSets);

for scale = 2:4
    fid = fopen(sprintf('PSNR_SSIM_scale=%d.txt', scale), 'w');
    scaleDir = strcat('X', num2str(scale));
    for iTestSet = 1:len
        testSet = testSets{1, iTestSet};
        fprintf(fid, '\n\n[Test set: %s]\n', testSet);
        Imgs = dir(fullfile(apath, 'GT', testSet, '*.png'));

        for idxImg = 1:length(Imgs)
            imgName = Imgs(idxImg).name;
            fprintf(fid, '\n[%d / %d] %s:\t\t', idxImg, length(Imgs), imgName);

            GT = imread(fullfile(apath, 'GT', testSet, imgName));
            if length(size(GT)) == 3
                GT = rgb2ycbcr(GT);
                GT = GT(:,:,1);
            end

            for iSR = 1:length(works)
                work = works{1, iSR};
                imgPath = fullfile(apath, work, testSet, scaleDir, imgName);
                file_missing = false;
                try
                    SR = imread(imgPath);
                catch
                    % Some files are missing
                    [hGT, wGT] = size(GT);
                    SR = im2uint8(rand(hGT, wGT, 3));
                    file_missing = true;
                end
                if length(size(SR)) == 3
                    SR = rgb2ycbcr(SR);
                    SR = SR(:,:,1);
                end
                [h,w,c] = size(SR);
                if ~strcmp(work, 'Aplus')
                    GT_ = GT(1:h, 1:w, :);
                    shave = scale;
                    GTc = GT_((1+shave):(h-shave), (1+shave):(w-shave));
                    SR = SR((1+shave):(h-shave), (1+shave):(w-shave));
                else
                    GTc = GT((1+scale):(h+scale), (1+scale):(w+scale));
                    % In Aplus, SR output is already shaved.
                end

                PSNR = psnr(GTc, SR);
                SSIM = ssim(GTc, SR);   
                if file_missing
                    PSNR = -1;
                    SSIM = -1;
                end
                fprintf(fid, '%s (%.2f / %.4f)   ', work, PSNR, SSIM);
            end
        end
    end
end