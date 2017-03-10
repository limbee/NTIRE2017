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
-- average H,W = 1435, 1971
------------------------------------------------------------------


local numImg = 800
local mean = torch.Tensor(3,1,1):zero():float()
local var = torch.Tensor(3,1,1):zero():float()
local height, width = 0,0
local maxHeight, maxWidth = 0,0
local argmaxHeight, argmaxWidth = 0,0 -- amh,amw = argmax_{h,w} (Area)
local maxArea = 0
local numPixels = 0

local function getName(i)
    if i % 100 == 0 then print(i) end
    local dirPath = '/var/tmp/dataset/DIV2K/DIV2K_train_HR'
    local filename = i
    local digit = i
    while digit < 1000 do
        filename = '0' .. filename
        digit = digit * 10
    end
    return paths.concat(dirPath, filename .. '.png')
end

for i=1,numImg do
    local filename = getName(i)
    local img = image.load(filename, 3, 'float')

    mean:add(img:sum(2):sum(3):squeeze())

    local h, w = img:size(2), img:size(3)
    local area = h * w
    height = height + h
    width = width + w

    if maxArea < area then
        maxArea = area
        argmaxHeight = h
        argmaxWidth = w
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
local avgH, avgW = height/800, width/800

print(('Max H, W = %d %d'):format(maxHeight, maxWidth))
print(('Average H, W = %d, %d'):format(avgH, avgW))
print(('Max area = %d (h,w = %d,%d)'):format(maxArea, argmaxHeight, argmaxWidth))
print(('Mean R,G,B = %.4f, %.4f, %.4f'):format(mean[1], mean[2], mean[3]))

print('\nCalculating variance...')
mean = mean:reshape(3,1,1)
for i=1,numImg do
    local filename = getName(i)
    local img = image.load(filename, 3, 'float')

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