torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('psnr.t7 converter for old experiments')
cmd:text()
cmd:text('Options:')
cmd:option('-fileName',     '',     'Files to convert (Delimiter is +)')
cmd:option('-interval',     1e3,    'X axis scale')

local opt = cmd:parse(arg or {})
opt.fileName = opt.fileName:split('+')

for i = 1, #opt.fileName do
    local tbl = torch.load(opt.fileName[i])
    local converted = {}
    if type(tbl[1]) == 'number' then
        local convertedSet = {}
        for j = 1, #tbl do
            table.insert(convertedSet, {key = j * opt.interval, value = tbl[j]})
        end
        table.insert(converted, convertedSet)
    elseif type(tbl[1]) == 'table' then
        for j = 1, #tbl do
            local convertedSet = {}
            for k = 1, #tbl[j] do
                table.insert(convertedSet, {key = j * opt.interval, value = tbl[j][k]})
            end
            table.insert(converted, convertedSet)
        end
    end
    torch.save(opt.fileName[i] .. 'rv', converted)
end