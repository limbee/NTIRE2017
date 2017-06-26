require 'image'
require 'cutorch'

cmd = torch.CmdLine()
cmd:option('-apath',        '/dataset/SR_compare',              'Absolute path')
cmd:option('-set',          'val',                              'Test set')
cmd:option('-scale',        4,                                  'Scale')
cmd:option('-patchSize',    100,                                'Patch size')
cmd:option('-nSample',      300,                                'Number of randomly cropped samples')
cmd:option('-nSave',        30,                                 'Number of samples to be saved')
cmd:option('-nLoad',        1e9,                                'Limit the number of loaded images (for debug)')
cmd:option('-works',        'Aplus+SRCNN+VDSR+SRResNet+SRResNet_reproduce+Ours_Single+Ours_Multi',        'works to compare')
cmd:option('-reset',        'true',                             'Reset')

local opt = cmd:parse(arg or {})
opt.reset = opt.reset == 'true'
assert(opt.nSave <= opt.nSample)
torch.manualSeed(os.date('%s') * os.date('%m'))

local apath = opt.apath
local works = opt.works:split('+')
local set = opt.set
local scale = tonumber(opt.scale)
local patchSize = tonumber(opt.patchSize)
local shave = 5
local ext = '.png'
local savePath = paths.concat('cropped_random', set, 'patchSize=' .. patchSize)
if opt.reset then
    os.execute('rm -r ' .. savePath .. '/*')
end

local function mkdir(p)
    if not paths.dirp(p) then
        paths.mkdir(p)
    end
end
mkdir('cropped_random')
mkdir(paths.concat('cropped_random', set))
mkdir(savePath)
mkdir(paths.concat(savePath, 'gather'))

local dir_SRResNet = 'SRRResNet'
if scale ~= 4 or set == 'Urban100' or set == 'val' then
    dir_SRResNet = 'SRResNet_reproduce'
end

local timer = torch.Timer()
print('loading images...')
local names, SRResNet, EDSR = {}, {}, {}
for imgName in paths.iterfiles(paths.concat(apath, 'GT', set)) do
    if #EDSR >= opt.nLoad then break end
    local name = imgName:split('%.')[1]
    names[#names + 1] = name
    SRResNet[#SRResNet + 1] = image.load(paths.concat(apath, dir_SRResNet, set, 'X' .. scale, imgName), 3, 'byte')
    EDSR[#EDSR + 1] = image.load(paths.concat(apath, 'Ours_Single', set, 'X' .. scale, imgName), 3, 'byte')
end
assert(#names == #SRResNet and #names == #EDSR)
print('\t Elapsed time: ' .. timer:time().real)

print('processing...')
for idx = 1, #EDSR do
    print(idx, names[idx])
    -- print('calculating gradients...')
    timer:reset()
    local diffs = torch.zeros(opt.nSample)
    local iSample_to_iFile = torch.Tensor(opt.nSample) -- Hash: idx of sample -> idx of file name
    local locs = torch.zeros(opt.nSample, 2) -- (x, y) crop position
    for i = 1, opt.nSample do
        -- local idx = math.random(1, #EDSR) -- idx of file name
        local h, w = EDSR[idx]:size(2), EDSR[idx]:size(3)
        h = h - shave
        w = w - shave
        local x = math.random(1, w - patchSize + 1)
        local y = math.random(1, h - patchSize + 1)
        local patch_SRResNet = SRResNet[idx][{{}, {y, y + patchSize - 1}, {x, x + patchSize - 1}}]
        local patch_EDSR = EDSR[idx][{{}, {y, y + patchSize - 1}, {x, x + patchSize - 1}}]
        -- patch_SRResNet = patch_SRResNet:cuda()
        -- patch_EDSR = patch_EDSR:cuda()
        -- patch_SRResNet = 0.2126 * patch_SRResNet[1] + 0.7152 * patch_SRResNet[2] + 0.0722 * patch_SRResNet[3]
        -- patch_EDSR = 0.2126 * patch_EDSR[1] + 0.7152 * patch_EDSR[2] + 0.0722 * patch_EDSR[3]
        patch_SRResNet = image.rgb2y(patch_SRResNet:float())
        patch_EDSR = image.rgb2y(patch_EDSR:float())
        local sub = (patch_SRResNet - patch_EDSR):abs():view(-1)
        local tops, _ = sub:topk(0.3 * sub:numel(), 1, true) -- only significant differences affect the sorting
        sub = sub[sub:gt(tops:min())]
        -- sub = sub[sub:gt(sub:mean())]
        local diff = 0
        if sub:size():size() ~= 0 then
            diff = sub:mean()
        end
        -- local diff = sub[sub:gt(tops:min())]:mean()
        -- local diff = (patch_SRResNet - patch_EDSR):abs():mean()

        diffs[i] = diff
        iSample_to_iFile[i] = idx
        locs[i] = torch.Tensor({x, y})

        patch_SRResNet = nil
        patch_EDSR = nil
        if #EDSR % 100 == 0 then
            collectgarbage()
            collectgarbage()
        end
    end
    -- print('\t Elapsed time: ' .. timer:time().real)

    -- print('sorting...')
    local _, sorted_indicies = torch.sort(diffs, 1, true)

    local function save_name(rank, imgName, work, x, y, diff)
        local name = 'rank=' .. rank .. '_' .. 'diff=' .. math.floor(diff)
        name = name .. '_' .. set .. '_' .. imgName .. '_' .. work .. '_sc=' .. scale
        name = name .. '_x=' .. x .. '_y=' .. y .. '_ps=' .. patchSize
        -- local name = set .. '_' .. imgName .. '_rank=' .. rank .. '_diff=' .. math.floor(diff)
        -- name = name .. '_' .. work .. '_sc=' .. scale
        -- name = name .. '_x=' .. x .. '_y=' .. y .. '_ps=' .. patchSize       
        if work == 'gather' then
            return paths.concat(savePath, 'gather', name .. ext)
        else
            return paths.concat(savePath, name .. ext)
        end
    end

    -- print('saving...')
    -- timer:reset()
    local pad = torch.zeros(3, patchSize, 10)
    for i = 1, opt.nSave do
        local iSample = sorted_indicies[i]
        local diff = diffs[iSample]
        local iFile = iSample_to_iFile[iSample]
        local imgName = names[iFile]
        local left, top = locs[iSample][1], locs[iSample][2]
        local right, bottom = left + patchSize - 1, top + patchSize - 1

        print('diff: ' .. diff)
        if diff > 15 then
            local gt = image.load(paths.concat(apath, 'GT', set, imgName .. ext))
            local bicubic = image.load(paths.concat(apath, 'Bicubic', set, 'X' .. scale, imgName .. ext))
            gt = gt[{{}, {top, bottom}, {left, right}}]
            bicubic = bicubic[{{}, {top, bottom}, {left, right}}]
            image.save(save_name(i, imgName, 'GT', left, top, diff), gt)
            image.save(save_name(i, imgName, 'Bicubic', left, top, diff), bicubic)

            local gather = torch.cat({gt, pad, bicubic}, 3)

            for _, work in pairs(works) do
                if not ((work:find('SRResNet') and scale ~= 4) or (work == 'SRResNet' and (set == 'val' or set == 'Urban100'))) then
                    local sr
                    if work == 'SRResNet' then
                        sr = image.load(paths.concat(apath, work, set, 'X' .. scale, imgName .. '_SRResNet-MSE' .. ext))
                    else
                        sr = image.load(paths.concat(apath, work, set, 'X' .. scale, imgName .. ext))
                    end
                    sr = sr[{{}, {top, bottom}, {left, right}}]
                    image.save(save_name(i, imgName, work, left, top, diff), sr)
                    gather = torch.cat({gather, pad, sr}, 3)
                end
            end

            image.save(save_name(i, imgName, 'gather', left, top, diff), gather)
        end
    end
    print('\t Elapsed time: ' .. timer:time().real .. '\n')
end