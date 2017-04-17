clear;
outputDir = 'img_output';
%outputDir = 'img_input';
targetDir = 'img_target';
%setException = {'Set5', 'Set14', 'B100', 'Urban100'};
psnrOnly = true;

tableRow = {};
tableCol = {};
tableData = [];
disp('--------------------------------------------------------------------------------')
disp('----------------------NTIRE2017: SNU-CVLAB evaluation tool----------------------')
disp('--------------------------------------------------------------------------------')
disp(' ')
disp([sprintf('%-25s', 'Model Name'), ' | ', ...
sprintf('%-10s', 'Set Name'), ' | ', ...
sprintf('%-5s', 'Scale'), ...
' | PSNR / SSIM'])

disp('--------------------------------------------------------------------------------')
totalDir = dir(fullfile(outputDir));
for iModel = 1:length(totalDir)
    modelName = totalDir(iModel).name;
    if modelName(1) == '.'
        continue;
    end
    modelFull = fullfile(outputDir, modelName);
    modelDir = dir(modelFull);
    row = 1;
    for iSet = 1:length(modelDir)
        setName = modelDir(iSet).name;
        if (setName(1) == '.') || (any(strcmp(setException, setName)) == true)
            continue;
        end
        setFull = fullfile(modelFull, setName);
        setDir = dir(setFull);
        for ix = 1:length(setDir)
            scaleName = setDir(ix).name;
            if scaleName(1) == '.'
                continue;
            end
            scale = str2num(scaleName(2:length(scaleName)));
            scaleFull = fullfile(setFull, scaleName);
            scaleDir = dir(scaleFull);
            meanPSNR = 0;
            meanSSIM = 0;
            numImages = 0;
            for im = 1:length(scaleDir)
                imageName = scaleDir(im).name;
                inputName = fullfile(scaleFull, imageName);
                targetName = fullfile(targetDir, modelName, setName, imageName);
                if (imageName(1) ~= '.') && (strcmp(imageName, 'Thumbs.db') == 0) && (exist(targetName, 'file') == 2)
                    inputImg = imread(inputName);
                    targetImg = imread(targetName);
                    targetDim = length(size(targetImg));
                    if targetDim == 2
                        targetImg = cat(3, targetImg, targetImg, targetImg);
                    end
                    shave = scale + 6;
                    [h, w, ~] = size(inputImg);
                    targetImg = targetImg(1:h, 1:w, :);
                    inputImg = inputImg((1 + shave):(h - shave), (1 + shave):(w - shave), :);
                    targetImg = targetImg((1 + shave):(h - shave), (1 + shave):(w - shave), :);
                    meanPSNR = meanPSNR + psnr(inputImg, targetImg);
                    if psnrOnly == false
                        meanSSIM = meanSSIM + ssim(inputImg, targetImg);
                    end
                    numImages = numImages + 1;
                end
                if (mod(im, 20) == 0) && (psnrOnly == false)
                    disp([num2str(im) '/' num2str(length(scaleDir))]);
                end
            end
            if (numImages > 0)
                meanPSNR = meanPSNR / numImages;
                meanSSIM = meanSSIM / numImages;
                if ix == 3
                    modelNameF = sprintf('%-25s', modelName);
                    setNameF = sprintf('%-10s', setName);
                    scaleF = sprintf('%-5d', scale);
                else
                    modelNameF = sprintf('%-25s', '');
                    setNameF = sprintf('%-10s', '');
                    scaleF = sprintf('%-5d', scale);
                end
                disp([modelNameF, ' | ', ...
                setNameF, ' | ', ...
                scaleF, ...
                ' | PSNR: ', num2str(meanPSNR, '%.3fdB')])
                if psnrOnly == false
                    disp([sprintf('%-25s', ''), ' | ', ...
                    sprintf('%-10s', ''), ' | ', ...
                    sprintf('%-5d', ''), ...
                    ' | SSIM: ', num2str(meanSSIM, '%.3f')])
                end
            end
        end
        disp('--------------------------------------------------------------------------------')
    end
end
