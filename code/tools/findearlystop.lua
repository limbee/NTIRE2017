require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cutorch'
require 'gnuplot'


trainName = "argx_unknown_x4"

a=torch.load("../experiment/".. trainName .. "/psnr.t7")
print(a)
b = 0
c = 0
for i =1, 300 do
    b = a[1][i].value
    
    if b > c then
    --if  a[i][3] > 31.37 then
        c = b
        d = i
    --end
    end
end
print(d)
print(c)
