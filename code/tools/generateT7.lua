require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('An image packing tool for DIV2K dataset')
cmd:text()
cmd:text('Options:')
cmd:option('-apath',        '/var/tmp/dataset',     'Absolute path of the DIV2K folder')
cmd:option('-scale',        '2_3_4',                'Scales to pack')
cmd:option('-split',        'false',                'split or pack')
cmd:option('-printEvery',   100,                    'print the progress # every iterations')

local opt = cmd:parse(arg or {})
opt.scale = opt.scale:split('_')
opt.split = (opt.split == 'true')
for i = 1, #opt.scale do
    opt.scale[i] = tonumber(opt.scale[i])
end

local targetPath = paths.concat(opt.apath, 'DIV2K')
local outputPath = paths.concat(opt.apath, 'DIV2K_decoded')

local hrDir = 'DIV2K_train_HR'
local lrDir =
{
    'DIV2K_train_LR_bicubic',
    'DIV2K_train_LR_unknown',
    'DIV2K_test_LR_bicubic',
    'DIV2K_test_LR_unknown'
}

if not paths.dirp(outputPath) then
    paths.mkdir(outputPath)
end

if not paths.dirp(paths.concat(outputPath, hrDir)) then
    paths.mkdir(paths.concat(outputPath, hrDir))
end

local convertTable = 
    {
        {
            tDir = paths.concat(targetPath, hrDir), 
            oDir = paths.concat(outputPath, hrDir)
        }
    }

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
            local img = image.load(fileDir):mul(255):byte()
            
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
