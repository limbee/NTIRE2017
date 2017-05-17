clear all;

absolPath = "/var/tmp/dataset/ILSVRC/";
dataPath = "Data/CLS-LOC/";
gtPath = "ImageSets/CLS-LOC/";
dataPathid = ["test/" "train/" "val/"];
gtPathid = ["test.txt" "train_cls.txt" "val.txt"];    

fid =fopen(strcat(absolPath, gtPath, 'random50k.txt'), 'w');
gtid= fopen(strcat(absolPath, gtPath, 'mergeAll.txt'),'r');
gtLength = 100000+1281167+50000;
targetLength = 50000;
rand('seed',1)
checkOrder = randperm(gtLength);

imagePath={};
for i = 1 : gtLength
    gt_i = fgetl(gtid);
    if gt_i==-1
        break;
    end
    gt_param = strsplit(gt_i, {' '});
    
    imagePath(i) = gt_param(1);   
end

j= 1;
for i=1:gtLength
    potenImage = imread(char(strcat(absolPath,dataPath,imagePath{checkOrder(i)},'.JPEG')));
    
    if size(size(potenImage),2) ==3
        if size(potenImage,1) >=192 && size(potenImage,2)>=192
            fprintf(fid, '%s %d %d %d %d\n',imagePath{checkOrder(i)},j,size(potenImage,1),size(potenImage,2),size(potenImage,3));
            if j ==targetLength
                break;
            end            
            j=j+1;
        end
    end
    
    
end

fclose(fid);

