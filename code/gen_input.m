folders = {'val','benchmark','91','291'}; % choose {'train','val','benchmark','91','291'};
makeSmall = true; % this is for SRResNet (C. Ledig et al., arXiv 2016)
makeBig = true; % this is for VDSR (J. Kim et al., CVPR 2016)

clearvars -except folders makeSmall makeBig

for i=1:length(folders)
    folder = folders{i};
	if strcmp(folder,'train') || strcmp(folder,'val')
		dataPath = fullfile('../dataset/ILSVRC2015/Data/CLS-LOC/',folder);
	else
		dataPath = fullfile('../dataset/',folder);
	end
	pathSmall = fullfile(dataPath,'small');
	pathBig = fullfile(dataPath,'big');

	if exist(pathSmall,'dir')==0 
		disp(['make save path ' pathSmall]);
		mkdir(pathSmall);
	end
	if exist(pathBig ,'dir')==0 
		disp(['make save path ' pathBig]);
		mkdir(pathBig);
	end

	dirList = dir(dataPath);
	dirList = dirList(~ismember({dirList.name},{'.','..','small','big'}));
	scale = 4;

	for iDir = 1:length(dirList)
		if strcmp(folder,'val') || strcmp(folder,'91') || strcmp(folder,'291') % they don't have sub directories
			if iDir==1 
                imgList = dirList;
				dirList = [1]; % dummy
				dirName = '';
            else
                continue;
			end
		else
			dirName = dirList(iDir).name;
			imgList = dir(fullfile(dataPath,dirName));
			imgList = imgList(~ismember({imgList.name},{'.','..'}));
        end
        
		if makeSmall
			subDirSmall = fullfile(pathSmall,dirName);
			if exist(subDirSmall,'dir')==0 
				mkdir(subDirSmall); 
			end
		end
		if makeBig
			subDirBig = fullfile(pathBig,dirName);
			if exist(subDirBig,'dir')==0 
				mkdir(subDirBig); 
			end
			for sc = 2:4
				subsubDirBig = fullfile(subDirBig,sprintf('x%d',sc));
				if exist(subsubDirBig,'dir')==0 
					mkdir(subsubDirBig); 
				end
			end
		end

		for iImg = 1 : length(imgList)
			fileName = fullfile(dataPath,dirName,imgList(iImg).name);
			disp(sprintf('[%d/%d][%d/%d] %d/%d: %s',i,length(folders),iDir,length(dirList),iImg,length(imgList),fileName));
			try
				image = imread(fileName);
			catch
				continue;
			end

			if ndims(image)==2 || (ndims(image)==3 && size(image,3)==1)
				image = cat(3,image,image,image);
			end
			image = im2double(image);

			[path,name,ext] = fileparts(fileName);
			sz_ = size(image);
			sz_ = sz_(1:2);
		
			if makeSmall
				sz = sz_-mod(sz_,scale);
				target = image(1:sz(1),1:sz(2),:);

				inputSmall = imresize(target,1/scale,'bicubic');
				inputSmall = im2uint8(inputSmall);
				newNameSmall = fullfile(subDirSmall,[name '.png']);
				imwrite(inputSmall,newNameSmall);
			end
			if makeBig
				for sc=2:4
					sz = sz_-mod(sz_,sc);
					target = image(1:sz(1),1:sz(2),:);			

					inputBig = imresize(imresize(target,1/sc,'bicubic'),sc,'bicubic');
					inputBig = im2uint8(inputBig);
					newNameBig = fullfile(subDirBig,sprintf('x%d',sc),[name '.png']);
					imwrite(inputBig,newNameBig);
				end
			end
		end
	end
end
