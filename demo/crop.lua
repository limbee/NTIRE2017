require 'image'

cmd = torch.CmdLine()
cmd:option('-apath',        '/var/tmp/dataset/DIV2K',   'Absolute path of dataset directory')
cmd:option('-idx',          1,                          'Index of input')
cmd:option('-scales',       '2_3_4',                    'Scales')
cmd:option('-lt',           '1_1',                      'left, top')
cmd:option('-wh',           '1_1',                      'width, height')
cmd:option('-ps',           '.',                        'square patch size')

local opt = cmd:parse(arg or {})

local apath = opt.apath
local scales = opt.scales:split('_')
local lt = opt.lt:split('_')
local wh = opt.wh:split('_')
local patchSize = opt.ps
if patchSize ~= '.' then
    patchSize = tonumber(patchSize)
    for i = 1, 2 do
        lt[i] = tonumber(lt[i])
        wh[i] = patchSize
    end
else
    for i = 1, 2 do
        lt[i] = tonumber(lt[i])
        wh[i] = tonumber(wh[i])
    end
    assert(wh[1] * wh[2] ~= 1)
end

local w, h = wh[1], wh[2]
local left, top = lt[1], lt[2]
local right, bottom = left + w - 1, top + h - 1

local ext = '.png'
local savePath = 'cropped'
if not paths.dirp(savePath) then
    paths.mkdir(savePath)
end

local function getFileName(idx)
    local digit = idx
    local fileName = idx
    while digit < 1000 do
        digit = digit * 10
        fileName = '0' .. fileName
    end
    return fileName
end

local imageName = getFileName(opt.idx)

local id = torch.random(1,1e6)
id = id .. '_' .. w .. 'x' .. h

for _,scale in pairs(scales) do
    -- local ilr_bic = image.load(paths.concat(
    --     apath, 'DIV2K_train_LR_bicubic', 'X' .. scale, imageName .. 'x' .. scale .. ext))
    -- local ilr_unk = image.load(paths.concat(
    --     apath, 'DIV2K_train_LR_unknown', 'X' .. scale, imageName .. 'x' .. scale .. ext))
    -- ilr_bic = image.scale(ilr_bic, scale)
    -- ilr_unk = image.scale(ilr_unk, scale)
    local ilr_bic = image.load(paths.concat(
        'img_interpolate', 'Interpolate_bicubic_' .. imageName .. 'x' .. scale .. ext))
    local ilr_unk = image.load(paths.concat(
        'img_interpolate', 'Interpolate_unknown_' .. imageName .. 'x' .. scale .. ext))
    local sr_bic = image.load(paths.concat(
        'img_output', 'bicubic_x' .. scale, 'val', 'X' .. scale, imageName .. ext))
    local sr_unk = image.load(paths.concat(
        'img_output', 'unknown_x' .. scale, 'val', 'X' .. scale, imageName .. ext))

    ilr_bic = ilr_bic[{{}, {top, bottom}, {left, right}}]
    ilr_unk = ilr_unk[{{}, {top, bottom}, {left, right}}]
    sr_bic = sr_bic[{{}, {top, bottom}, {left, right}}]
    sr_unk = sr_unk[{{}, {top, bottom}, {left, right}}]

    local ilr_bic_name = paths.concat('cropped', imageName .. '_' .. id .. '_ilr_bic_x' .. scale .. ext)
    local ilr_unk_name = paths.concat('cropped', imageName .. '_' .. id .. '_ilr_unk_x' .. scale .. ext)
    local sr_bic_name = paths.concat('cropped', imageName .. '_' .. id .. '_sr_bic_x' .. scale .. ext)
    local sr_unk_name = paths.concat('cropped', imageName .. '_' .. id .. '_sr_unk_x' .. scale .. ext)

    image.save(ilr_bic_name, ilr_bic)
    image.save(ilr_unk_name, ilr_unk)
    image.save(sr_bic_name, sr_bic)
    image.save(sr_unk_name, sr_unk)
end

local hr = image.load(paths.concat(
    apath, 'DIV2K_train_HR', imageName .. ext))
hr = hr [{{}, {top, bottom}, {left, right}}]
local hr_name = paths.concat('cropped', imageName .. '_' .. id .. '_hr' .. ext)
image.save(hr_name, hr)