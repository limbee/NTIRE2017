clear;
outputDir = 'img_output';
%outputDir = 'img_input';
targetDir = 'img_target';
setException = {};
psnrOnly = true;

tableRow = {};
tableCol = {};
tableData = [];
col = 1;
row = 1;

totalDir = dir(fullfile(outputDir));
for iModel = 1:length(totalDir)
    modelName = totalDir(iModel).name;
    if (modelName(1) == '.')
        continue;
    end
    tableRow = [tableRow {[modelName '_PSNR']}];
    if (psnrOnly == false)
        tableRow = [tableRow {[modelName '_SSIM']}];
    end
    modelFull = fullfile(outputDir, modelName);
    modelDir = dir(modelFull);
    row = 1;
    for iSet = 1:length(modelDir)
        setName = modelDir(iSet).name;
        if ((setName(1) == '.') || (any(strcmp(setException, setName)) == true))
            continue;
        end
        setFull = fullfile(modelFull, setName);
        setDir = dir(setFull);
        for ix = 1:length(setDir)
            scaleName = setDir(ix).name;
            if (scaleName(1) == '.')
                continue;
            end
            scaleFull = fullfile(setFull, scaleName);
            scaleDir = dir(scaleFull);
            meanPSNR = 0;
            meanSSIM = 0;
            numImages = 0;
            disp(['Evaluate ' modelName ' on ' setName ' ' scaleName]);
            for im = 1:length(scaleDir)
                imageName = scaleDir(im).name;
                inputName = fullfile(scaleFull, imageName);
                targetName = fullfile(targetDir, modelName, setName, imageName);
                if ((imageName(1) ~= '.') && (strcmp(imageName, 'Thumbs.db') == 0) && (exist(targetName, 'file') == 2))
                    inputImg = imread(inputName);
                    targetImg = imread(targetName);
                    targetDim = length(size(targetImg));
                    if (targetDim == 2)
                        targetImg = cat(3, targetImg, targetImg, targetImg);
                    end
                    scale = 2;
                    shave = scale + 6;
                    imgSize = size(inputImg);
                    targetImg = targetImg(1:imgSize(1), 1:imgSize(2), :);
                    inputImg = inputImg((1 + shave):(imgSize(1) - shave), (1 + shave):(imgSize(2) - shave), :);
                    targetImg = targetImg((1 + shave):(imgSize(1) - shave), (1 + shave):(imgSize(2) - shave), :);
                    meanPSNR = meanPSNR + psnr(inputImg, targetImg);
                    if (psnrOnly == false)
                        meanSSIM = meanSSIM + ssim(inputImg, targetImg);
                    end
                    numImages = numImages + 1;
                end
                if ((mod(im, 20) == 0) && (psnrOnly == false))
                    disp([num2str(im) '/' num2str(length(scaleDir))]);
                end
            end
            if (numImages > 0)
                if (col == 1)
                    tableCol = [tableCol {[setName ' ' scaleName]}];
                end
                meanPSNR = meanPSNR / numImages;
                meanSSIM = meanSSIM / numImages;
                tableData(row, col) = meanPSNR;
                if (psnrOnly == false)
                    tableData(row, col + 1) = meanSSIM;
                end
                row = row + 1;
            end
        end
    end
    col = col + 1;
    if (psnrOnly == false)
        col = col + 1;
    end
end
T = array2table(tableData, 'RowNames', tableCol, 'VariableNames', tableRow);
disp('');
disp(T);
