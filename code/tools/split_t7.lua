local apath = '../../../var/tmp/dataset/DIV2K'
local setName = 'DIV2K'
local ref = 'train_HR'
local LR = {'train_LR_bicubic', 'train_LR_unknown', 'valid_LR_bicubic', 'valid_LR_unknown'}
local scale = {2, 3, 4}

local convertTable = {}
local dirTable = {}
local scaleTable = {}
table.insert(convertTable, setName .. '_' .. ref .. '.t7')
table.insert(dirTable, setName .. '_' .. ref)
table.insert(scaleTable, '')
for i = 1, #LR do
	for j = 1, #scale do
		table.insert(convertTable, setName .. '_' .. LR[i] .. '_X' .. scale[j] .. '.t7')
		table.insert(dirTable, setName .. '_' .. LR[i] .. '/X' .. scale[j])
		table.insert(scaleTable, 'x' .. scale[j])
		if (string.find(LR[i], 'bicubic')) then
			table.insert(convertTable, setName .. '_' .. LR[i] .. '_X' .. scale[j] .. 'b.t7')
			table.insert(dirTable, setName .. '_' .. LR[i] .. '/X' .. scale[j] .. 'b')
			table.insert(scaleTable, 'x' .. scale[j])
		end
	end
end

for i = 1, #convertTable do
	print('Converting ' .. convertTable[i])
	local image = torch.load(apath .. '/DIV2K_decoded/' .. convertTable[i])
	collectgarbage()
	local tDir = apath .. '/' .. dirTable[i]
	if (not paths.dirp(tDir)) then
		path.mkdir(tDir)
	end
	local offset = 0
	if (string.find(convertTable[i], 'valid')) then
		offset = 800
	end
	for j = 1, #image do
		local idx = j + offset
		local zp = nil
		if (idx < 10) then
			zp = '000'
		elseif (idx < 100) then
			zp = '00'
		else
			zp = '0'
		end
		torch.save(tDir .. '/' .. zp .. idx .. scaleTable[i] .. '.t7', image[j])
	end
end
