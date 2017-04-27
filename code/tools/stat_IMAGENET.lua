require 'image'

------------------------------------------------------------------
-- DIV2K dataset
----  Mean, Std
----  Height, Width, Area, etc.
--
-- [Result]
-- average [R,G,B] = [0.4488, 0.4371, 0.4040]
-- std [R,G,B] = [0.2845, 0.2701, 0.2920]
-- max H,W = 2040, 2040 (max Area is also 2040*2040)
-- In the range of 1~800, following images have the same size (2040*2040)
-- 0044, 0155, 0217, 0450, 0500, 0513, 0716
-- average H,W = 1435, 1971
------------------------------------------------------------------


local numImg = 50000
local mean = torch.Tensor(3,1,1):zero():float()
local var = torch.Tensor(3,1,1):zero():float()
local height, width = 0,0
local maxHeight, maxWidth = 0,0
local argmaxHeight, argmaxWidth = 0,0 -- amh,amw = argmax_{h,w} (Area)
local maxArea = 0
local numPixels = 0
local maxImg, maxImgsTable = '', {}

local function getName(i)
    if i % 100 == 0 then print(i) end
    local dirPath = '/var/tmp/dataset/IMAGENET_decoded/IMAGENET_HR'
    local filename = i
    local digit = i
    while digit < 10000 do
        filename = '0' .. filename
        digit = digit * 10
    end
    return paths.concat(dirPath, filename .. '.t7')
end

for i=1,numImg do
    local filename = getName(i)
    local img = torch.load(filename):float():div(255)

    mean:add(img:sum(2):sum(3):squeeze())

    local h, w = img:size(2), img:size(3)
    local area = h * w
    height = height + h
    width = width + w

    if maxArea < area then
        maxArea = area
        argmaxHeight = h
        argmaxWidth = w
        maxImgsTable = {}
        maxImg = filename
        table.insert(maxImgsTable, maxImg)
    elseif maxArea == area then
        table.insert(maxImgsTable, filename)
    end

    if maxHeight < h then
        maxHeight = h
    end
    if maxWidth < w then
        maxWidth = w
    end

    numPixels = numPixels + area

    img = nil
    collectgarbage()
    collectgarbage()
end

mean = mean:squeeze()
mean:div(numPixels)
local avgH, avgW = height/50000, width/50000

print(('Max H, W = %d %d'):format(maxHeight, maxWidth))
print(('Average H, W = %d, %d'):format(avgH, avgW))
print(('Max area = %d (h,w = %d,%d)'):format(maxArea, argmaxHeight, argmaxWidth))
print(('Mean R,G,B = %.4f, %.4f, %.4f'):format(mean[1], mean[2], mean[3]))
print(('%s has the maximum area: %d'):format(maxImg, maxArea))
print('Following files have the save maximum area')
print(maxImgsTable)

print('\nCalculating variance...')
mean = mean:reshape(3,1,1)
for i=1,numImg do
    local filename = getName(i)
    local img = torch.load(filename):float():div(255)

    img:add(-1, mean:repeatTensor(1,img:size(2),img:size(3)))
    var:add(1, img:pow(2):sum(2):sum(3))

    img = nil
    collectgarbage()
    collectgarbage()
end

var = var:squeeze()
var:div(numPixels)
local std = var:sqrt()

print(('Std R,G,B = %.4f, %.4f, %.4f'):format(std[1], std[2], std[3]))
