require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-apath',        '/var/tmp/dataset',     'Absolute path of the DIV2K folder')
cmd:option('-type',         'DIV2K_train_HR',             'Choose type of t7pack file to test')
cmd:option('-save',         '../../test_t7pack',       'Save path for images')
local opt = cmd:parse(arg or {})

local t7pack = torch.load(paths.concat(opt.apath, 'DIV2K_decoded', opt.type, 'pack.t7'))
local savepath = paths.concat(opt.save, opt.type)
if not paths.dirp(opt.save) then
    paths.mkdir(opt.save)
end
if not paths.dirp(paths.concat(opt.save, opt.type)) then
    paths.mkdir(paths.concat(opt.save, opt.type))
end

for i = 1, #t7pack do
    if i % 100 == 0 then print(i) end
    local img = t7pack[i]
    image.save(paths.concat(savepath, i .. '.jpg'), img)
end