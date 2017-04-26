require 'image'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('An image packing tool for DIV2K dataset')
cmd:text()
cmd:text('Options:')
cmd:option('-apath',        '/var/tmp/dataset',     'Absolute path of the DIV2K folder')
cmd:option('-dataset',      'DIV2K',                'Dataset to convert: DIV2K | Flickr2K')
cmd:option('-scale',        '2_3_4',                'Scales to pack')
cmd:option('-split',        'true',                 'split or pack')
cmd:option('-augment',      'true',                 'pre-augment dataset, useful for unknown downsampling')
cmd:option('-hr',           'true',                 'Generate HR data')
cmd:option('-lr',           'true',                 'Generate LR data')
cmd:option('-lrAug',        'true',                 'Generate pre-augmented unknown LR data')
cmd:option('-printEvery',   100,                    'print the progress # every iterations')

local opt = cmd:parse(arg or {})
opt.scale = opt.scale:split('_')
opt.split = opt.split == 'true'
opt.hr = opt.hr == 'true'
opt.lr = opt.lr == 'true'
opt.lrAug = opt.lrAug == 'true'
for i = 1, #opt.scale do
    opt.scale[i] = tonumber(opt.scale[i])
end

local targetPath, outputPath
local hrDir, lrDir

local targetPath = paths.concat(opt.apath, opt.dataset)
local outputPath = paths.concat(opt.apath, opt.dataset .. '_decoded')

if opt.dataset == 'DIV2K' then
    hrDir = 'DIV2K_train_HR'
    lrDir = {}
    if opt.lr then
        lrDir =  
        {
            'DIV2K_train_LR_bicubic',
            'DIV2K_train_LR_unknown',
            'DIV2K_test_LR_bicubic',
            'DIV2K_test_LR_unknown'
        }
    end
    if opt.lrAug then
        table.insert(lrDir, 'DIV2K_train_LR_unknown_augment')
    end
elseif opt.dataset == 'Flickr2K' then
    hrDir = 'Flickr2K_HR'
    lrDir = {}
    if opt.lr then
        lrDir =
        {
            'Flickr2K_LR_bicubic',
            'Flickr2K_LR_unknown'
        }
    end
    if opt.lrAug then
        table.insert(lrDir, 'Flickr2K_LR_unknown_augment')
    end
elseif opt.dataset == 'IMAGENET' then
    hrDir = 'IMAGENET_HR'
    lrDir = {}
    if opt.lr then
        lrDir =
        {
            'IMAGENET_LR_bicubic',
            --'IMAGENET_LR_unknown'
        }
    end
    if opt.lrAug then
        table.insert(lrDir, 'IMAGENET_LR_unknown_augment')
    end
end

if not paths.dirp(outputPath) then
    paths.mkdir(outputPath)
end

if not paths.dirp(paths.concat(outputPath, hrDir)) then
    paths.mkdir(paths.concat(outputPath, hrDir))
end

local convertTable = {}
if opt.hr then
    table.insert(convertTable,
        {tDir = paths.concat(targetPath, hrDir), 
        oDir = paths.concat(outputPath, hrDir)})
end

for i = 1, #lrDir do
    for j = 1, #opt.scale do
        local targetDir = paths.concat(targetPath, lrDir[i], 'X' .. opt.scale[j])
        local outputDir = paths.concat(outputPath, lrDir[i], 'X' .. opt.scale[j])
        if paths.dirp(targetDir) then
            if not paths.dirp(outputDir) then
                paths.mkdir(outputDir)
            end
            table.insert(convertTable, {tDir = targetDir, oDir = outputDir})
        end
    end
end

local ext = '.png'
for i = 1, #convertTable do
    print('Converting ' .. convertTable[i].tDir)
    
    local imgTable = {}
    local n = 0
    local fileList = paths.dir(convertTable[i].tDir)
    table.sort(fileList)
    for j = 1, #fileList do
        if fileList[j]:find(ext) then
            local fileDir = paths.concat(convertTable[i].tDir, fileList[j])
            local img = image.load(fileDir, 3, 'byte')
            
            if opt.split then
                local fileName = fileList[j]:split('.png')[1] .. '.t7'
                torch.save(paths.concat(convertTable[i].oDir, fileName), img)
            else
                table.insert(imgTable, img)
            end

            n = n + 1
            if ((n % opt.printEvery) == 0) then
                print('Converted ' .. n .. ' files')
            end
        end
    end

    if not opt.split then
        torch.save(paths.concat(convertTable[i].oDir, 'pack.t7'), imgTable)
    end

    imageTable = nil
    collectgarbage()
    collectgarbage()
end
