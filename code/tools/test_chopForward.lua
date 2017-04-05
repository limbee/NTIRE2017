require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-model',        '',         'Absolute path of the model')
cmd:option('-degrade',      'bicubic',  'Degrading operator to test')
cmd:option('-gpuid',        1,          'GPU ID for test')
cmd:option('-chopShave',     30,         'chopForward shave')
cmd:option('-chopMaxSize',  90000,      'chopForward max size')
cmd:option('-testRec',      'true',     'Whether test recursiveForward or not')
local opt = cmd:parse(arg or {})
opt.testRec = opt.testRec == 'true'

local gt = image.load('/var/tmp/dataset/DIV2K/DIV2K_train_HR/0010.png')
local testTable = {}
--table.insert(testTable, {image.load('/var/tmp/dataset/DIV2K/DIV2K_train_LR_' .. opt.degrade .. '/X2/0010x2.png'), 2})
--table.insert(testTable, {image.load('/var/tmp/dataset/DIV2K/DIV2K_train_LR_' .. opt.degrade .. '/X3/0010x3.png'), 3})
table.insert(testTable, {image.load('/var/tmp/dataset/DIV2K/DIV2K_train_LR_' .. opt.degrade .. '/X4/0010x4.png'), 4})

local util = require ('../utils')(opt)

if opt.model ~= '' then
    local model = torch.load(opt.model)
    for i = 1, #testTable do
        print('Scale X' .. testTable[i][2])
        testTable[i][1] = nn.Unsqueeze(1):forward(testTable[i][1]):cuda():mul(255)

        local timer_chop = torch.Timer()
        local output_chop = util:chopForward(testTable[i][1], model, testTable[i][2])
        util:quantize(output_chop, 255)
        print('chopForward - Test time:\t' .. timer_chop:time().real)
        psnr_chop = util:calcPSNR(output_chop:squeeze(1), gt:cuda(), testTable[i][2])
        print('chopForward - PSNR:\t\t' .. psnr_chop .. 'dB')

        model:clearState()
        collectgarbage()
        collectgarbage()

        if opt.testRec then
            local timer_rec = torch.Timer()
            local output_rec = util:recursiveForward(testTable[i][1], model, false)
            util:quantize(output_rec, 255)
            print('recursiveForward - testTime:\t' .. timer_rec:time().real)
            psnr_rec = util:calcPSNR(output_rec:squeeze(1), gt:cuda(), testTable[i][2])
            print('recursiveForward - PSNR:\t' .. psnr_rec .. 'dB')

            model:clearState()
            collectgarbage()
            collectgarbage()
        end
    end
end