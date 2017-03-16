require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local apath = '/var/tmp/dataset/DIV2K'
local setName = 'DIV2K'
local ref = 'train_HR'
--local LR = {'train_LR_bicubic', 'train_LR_unknown', 'valid_LR_bicubic', 'valid_LR_unknown'}
local LR = {'train_LR_bicubic', 'valid_LR_bicubic'}
local scale = {2, 3, 4}

local convertTable = {}
local dirTable = {}
table.insert(convertTable, setName .. '_' .. ref)
--table.insert(dirTable, setName .. '_' .. ref)
table.insert(dirTable, setName .. '_' .. ref .. 'r')

for i = 1, #LR do
    for j = 1, #scale do
        table.insert(convertTable, setName .. '_' .. LR[i] .. '/' .. 'X' .. scale[j])
        table.insert(dirTable, 'DIV2K_decoded' .. '/' .. setName .. '_' .. LR[i] .. '_X' .. scale[j] .. 'r')
        --table.insert(dirTable, setName .. '_' .. LR[i] .. '_X' .. scale[j])
        --table.insert(convertTable, setName .. '_' .. LR[i] .. '/X' .. scale[j] .. 'b')
        --table.insert(dirTable, setName .. '_' .. LR[i] .. '_X' .. scale[j] .. 'b')
    end
end

--local ext = 'png'
local ext = 'r.png'
for i = 1, #convertTable do
    print('Converting ' .. convertTable[i])
    local imageTable = {}
    local abscvt = apath .. '/' .. convertTable[i]
    local n = 0
    for file in paths.files(abscvt) do
        local fileDir = abscvt .. '/' .. file
        if (file:find(ext)) then
            local img = image.load(fileDir)
            table.insert(imageTable, img:mul(255):byte())
            n = n + 1
            if ((n % 100) == 0) then
                print('Converted ' .. n .. ' files')
            end
        end
    end
    torch.save(apath .. '/' .. dirTable[i] .. '.t7', imageTable)
    imageTable = nil
    collectgarbage()
end
