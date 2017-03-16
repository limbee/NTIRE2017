require 'nn'
require 'cunn'
require 'cudnn'
require 'tvnorm-nn'
require 'image'
require 'gnuplot'

local opt =
{
    batchSize = 1,
    nChannel = 3,
    patchSize = 96,
    gradPower = 2,
    numData = 800,
    numVal = 10,
    samples = 10000
}

local gradnet = nn.SpatialTVNormCriterion():cuda()
local usnet = nn.Unsqueeze(1):cuda()
local dataDir = '/var/tmp/dataset/DIV2K/DIV2K_train_LR_bicubic/X2'

local list = {}

for i = 1, opt.samples do
    if i % 100 == 0 then
        print('Iter ' .. i .. '/' .. opt.samples)
    end

    local idx = torch.random(1, opt.numData - opt.numVal)
    local idxStr = idx
    if idx < 10 then
        idxStr = '000' .. idx
    elseif idx < 100 then
        idxStr = '00' .. idx
    else
        idxStr = '0' .. idx
    end
    local img = image.load(paths.concat(dataDir, idxStr .. 'x2.png')):cuda()
    local c, h, w = table.unpack(img:size():totable())

    local rH = torch.random(1, h - opt.patchSize + 1)
    local rW = torch.random(1, w - opt.patchSize + 1)

    local batch = img[{{}, {rH, rH + opt.patchSize - 1}, {rW, rW + opt.patchSize - 1}}]
    batch = usnet:forward(batch):clone()
    local grad = gradnet:forward(batch)
    batch = batch:squeeze(1)    
    image.save('gradientSamples/' .. idx .. '_' .. rW .. '_' .. rH .. '_' .. grad .. '.png', batch)
    table.insert(list, {G = grad, Index = idx, x = rW, y = rH})
end

table.sort(list, function(a, b) return a.G > b.G end)

torch.save('gradientSamples/gradientStatistics.t7', list)

print('Top 100:')
for i = 1, 100 do
    print(list[i].G)
end

print('Show percentage:')
for i = 1, 10 do
    print(list[opt.samples * i / 10])
end

local hist = torch.Tensor(opt.samples)
for i = 1, opt.samples do
    hist[i] = list[i].G
end

gnuplot.pdffigure('Histogram.pdf')
gnuplot.hist(hist, 20)
gnuplot.plotflush()
