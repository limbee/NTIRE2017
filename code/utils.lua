require 'nn'

local M = {}
local util = torch.class('sr.util',M)

function util:__init(opt)
    if opt then
        self.opt = opt
        self.save = opt.save
    end
end

function util:plot(tb,name)
    local fig = gnuplot.pdffigure(paths.concat(self.save,name .. '.pdf'))
    local lines = {}
    if torch.type(tb[1]):find('Tensor') then
        local nLine = tb[1]:size(1)
        local value = {}
        for i=1,nLine do 
            value[i] = torch.Tensor(#tb)
            for j=1,#tb do
                value[i][j] = tb[j][i]
            end
            table.insert(lines, {name..' x'..tostring(i+1), value[i],'-'})
        end
    else
        table.insert(lines,{name,torch.Tensor(tb),'-'})
    end
    gnuplot.plot(lines)
    gnuplot.grid(true)
    gnuplot.title(name)
    gnuplot.xlabel('iteration (*' .. self.opt.testEvery .. ')')
    if torch.type(tb[1]):find('Tensor') then
        if tb[1][1] < tb[#tb][1] then
            gnuplot.movelegend('right','bottom')
        else
            gnuplot.movelegend('right','top')
        end
    else
        if tb[1] < tb[#tb] then
            gnuplot.movelegend('right','bottom')
        else
            gnuplot.movelegend('right','top')
        end
    end
	gnuplot.plotflush(fig)
	gnuplot.closeall()  
end

function util:checkpoint(model,criterion,loss,psnr)
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    model:clearState()

    torch.save(paths.concat(self.save,'model','model_' .. #loss .. '.t7'),model)
    torch.save(paths.concat(self.save,'model','model_latest.t7'),model)

    torch.save(paths.concat(self.save,'loss.t7'),loss)
    torch.save(paths.concat(self.save,'psnr.t7'),psnr)
    torch.save(paths.concat(self.save,'opt.t7'),self.opt)
end

function util:load()
    local ok, loss, psnr
    if self.opt.load then
        ok,loss,psnr,opt = pcall(
	function()
	    local loss = torch.load(paths.concat(self.save,'loss.t7'))
	    local psnr = torch.load(paths.concat(self.save,'psnr.t7'))
	    local opt = torch.load(paths.concat(self.save,'opt.t7'))
	    return loss,psnr,opt
	end)
        if ok then
            print(('loaded history (%d epoch * %d iter/epoch)\n'):format(#loss,self.opt.testEvery))
        else
            print('history (loss, psnr, options) does not exist')
            loss, psnr = {},{}
        end
    else
        ok = false
        loss, psnr = {},{}
    end

    return ok, loss, psnr
end

function util:calcPSNR(output,target,scale)
    output = output:squeeze()
    target = target:squeeze()
    if output:dim()==3 then
        output = self:rgb2y(output:float())
        target = self:rgb2y(target:float())
    end

    local h,w = table.unpack(output:size():totable())
    local sc = scale 
    local diff = (output - target)[{{sc + 1, h - sc}, {sc + 1, w - sc}}]
    local mse = diff:pow(2):mean()
    local psnr = 10*math.log10(255*255/mse)

    return psnr
end

function util:rgb2y(img)
    local y = img.new():resize(img:size(2),img:size(3)):fill(16)
    y:add(img[1] * 65.738 / 256)
    y:add(img[2] * 129.057 / 256)
    y:add(img[3] * 25.064 / 256)
    y:clamp(16,235)
    return y
end

return M.util
