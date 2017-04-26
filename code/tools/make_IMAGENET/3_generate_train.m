clear all;

absolPath = "/var/tmp/dataset/ILSVRC/";
dataPath = "Data/CLS-LOC/";
gtPath = "ImageSets/CLS-LOC/";
dataPathid = ["test/" "train/" "val/"];
gtPathid = ["test.txt" "train_cls.txt" "val.txt"];    

newAbsolPath ="/var/tmp/dataset/IMAGENET/";
newDataPath = ["IMAGENET_HR/" "IMAGENET_LR_bicubic/"];
newDataPathid= ["X2/" "X3/" "X4/"];

gtid= fopen(strcat(absolPath, gtPath, 'random50k.txt'),'r');
targetLength = 50000;
imagePath={};
for i = 1 : targetLength
    gt_i = fgetl(gtid);
    if gt_i==-1
        break;
    end
    gt_param = strsplit(gt_i, {' '});
    
    potenImage = imread(char(strcat(absolPath,dataPath,gt_param(1),'.JPEG')));
    filename=getFilename(i);
    
    imwrite(potenImage, char(strcat(newAbsolPath,newDataPath(1),filename,'.png')));
    for j=2:4
        lrPotenImage = imresize(potenImage, 1/j, 'bicubic');
        imwrite(lrPotenImage, char(strcat(newAbsolPath,newDataPath(2),newDataPathid(j-1),filename,'x',num2str(j),'.png')));
    end
    
    
end


function filename = getFilename(i)
    filename = num2str(i);
    digit = i;
    while (digit<10000)
       filename = strcat('0',filename);
       digit = digit*10;
    end
    
end

