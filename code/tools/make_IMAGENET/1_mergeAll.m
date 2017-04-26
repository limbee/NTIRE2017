clear all;

absolPath = "/var/tmp/dataset/ILSVRC/";
dataPath = "Data/CLS-LOC/";
gtPath = "ImageSets/CLS-LOC/";
dataPathid = ["test/" "train/" "val/"];
gtPathid = ["test.txt" "train_cls.txt" "val.txt"];    
        
fid= fopen(strcat(absolPath, gtPath, 'mergeAll.txt'),'w');
   
for i = 1 : 3
    gtid = fopen(strcat(absolPath,gtPath,gtPathid(i)), 'r');

    merge_GT(fid,gtid,dataPathid(i))
end

fclose(fid);

function merge_GT(fid,gtid,dataPathid)
    while(1)
        gt_i = fgetl(gtid);
        if gt_i==-1
            break;
        end
        fprintf(fid, '%s%s\n',dataPathid,gt_i);
        
    end
     
end
