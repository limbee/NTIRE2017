require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('A image packing tool for DIV2K dataset')
cmd:text()
cmd:text('Options:')
cmd:option('-apath',     '/var/tmp/dataset',    'Absolute path of the DIV2K folder')
cmd:option('-scale',     '1.5_2_3_4',           'Scales to pack')
cmd:option('-split',     'false',               'Split or pack')

local opt = cmd:parse(arg or {})
opt.apath = paths.concat(opt.apath, 'DIV2K')
opt.scale = opt.scale:split('_')
opt.split = (opt.split == 'true')
for i = 1, #opt.scale do
    opt.scale[i] = tonumber(opt.scale[i])
end

local hrDir = 'DIV2K_train_HR'
local lrDir =
{
    'DIV2K_train_LR_bicubic',
    'DIV2K_train_LR_unknown',
    'DIV2K_valid_LR_bicubic',
    'DIV2K_valid_LR_unknown'
}
local decDir = 'DIV2K_decoded'
if not paths.dirp(paths.concat(opt.apath, decDir)) then
    paths.mkdir(paths.concat(opt.apath, decDir))
end

local convertTable = {{scale = nil, dir = paths.concat(opt.apath, hrDir), saveAs = hrDir}}
for i = 1, #lrDir do
    for j = 1, #opt.scale do
        local targetDir = paths.concat(opt.apath, lrDir[i], 'X' .. opt.scale[j])
        if paths.dirp(targetDir) then
            table.insert(convertTable, {scale = opt.scale[j], dir = targetDir, saveAs = lrDir[i] .. '_X' .. opt.scale[j]})
        end
    end
end

local ext = '.png'
for i = 1, #convertTable do
    print('Converting ' .. convertTable[i].dir)
    
    local imgTable = {}
    local n = 0
    for file in paths.files(convertTable[i].dir) do
        local fileDir = paths.concat(convertTable[i].dir, file)
        if file:find(ext) then
            local img = image.load(fileDir):mul(255):byte()
            table.insert(imgTable, img)
            if opt.split then
                local fileName = file:split('.png')[1] .. '.t7'
                torch.save(paths.concat(opt.apath, convertTable[i].dir, fileName), img)
            end
            n = n + 1
            if ((n % 100) == 0) then
                print('Converted ' .. n .. ' files')
            end
        end
    end

    if not opt.split then
        torch.save(paths.concat(opt.apath, decDir, convertTable[i].saveAs .. '.t7'), imgTable)
    end

    imageTable = nil
    collectgarbage()
    collectgarbage()
end
