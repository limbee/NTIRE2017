clear;
outputDir = 'img_output';
targetDir = 'img_target';
modelException = {};
setException = {};
psnrOnly = true;

tableRow = {};
tableCol = {};
tableData = [];
disp(repmat('-', 1, 80))
disp([repmat('-', 1, 22), 'NTIRE2017: SNU-CVLAB evaluation tool', repmat('-', 1, 22)])
disp(repmat('-', 1, 80))
disp(' ')
disp([sprintf('%-25s', 'Model Name'), ' | ', ...
sprintf('%-10s', 'Set Name'), ' | ', ...
sprintf('%-5s', 'Scale'), ...
' | PSNR / SSIM'])

disp(repmat('-', 1, 80))
totalDir = dir(fullfile(outputDir));
for iModel = 1:length(totalDir)
    modelName = totalDir(iModel).name;
    if (modelName(1) == '.') || (any(strcmp(modelException, modelName)) == true)
        continue;
    end
    modelFull = fullfile(outputDir, modelName);
    modelDir = dir(modelFull);
    isModelPrint = false;
    for iSet = 1:length(modelDir)
        setName = modelDir(iSet).name;
        if (setName(1) == '.') || (any(strcmp(setException, setName)) == true)
            continue;
        end
        setFull = fullfile(modelFull, setName);
        setDir = dir(setFull);
        isSetPrint = false;
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
                    if sum(strcmp(setName, {'Set5', 'Set14', 'B100', 'Urban100'})) == 1 
                        targetImg = rgb2ycbcr(targetImg);
                        targetImg = targetImg(:,:,1);
                        inputImg = rgb2ycbcr(inputImg);
                        inputImg = inputImg(:,:,1);
						shave = scale;
                    end
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
            end
            if (numImages > 0)
                meanPSNR = meanPSNR / numImages;
                meanSSIM = meanSSIM / numImages;
                if isModelPrint == false
                    modelNameF = sprintf('%-25s', modelName);
                    setNameF = sprintf('%-10s', setName);
                    scaleF = sprintf('%-5d', scale);
                    isModelPrint = true;
                    isSetPrint = true;
                elseif isSetPrint == false
                    disp([repmat(' ', 1, 26), repmat('-', 1, 54)]);
                    modelNameF = repmat(' ', 1, 25);
                    setNameF = sprintf('%-10s', setName);
                    scaleF = sprintf('%-5d', scale);
		            isSetPrint = true;
                else
                    modelNameF = repmat(' ', 1, 25);
                    setNameF = repmat(' ', 1, 10);
                    scaleF = sprintf('%-5d', scale);
                end
                disp([modelNameF, ' | ', ...
                setNameF, ' | ', ...
                scaleF, ...
                ' | PSNR: ', num2str(meanPSNR, '%.2fdB')])
                if psnrOnly == false
                    disp([repmat(' ', 1, 25), ' | ', ...
                    repmat(' ', 1, 10), ' | ', ...
                    repmat(' ', 1, 5), ...
                    ' | SSIM: ', num2str(meanSSIM, '%.4f')])
                end
            end
        end
    end
    disp(repmat('-', 1, 80))
end
