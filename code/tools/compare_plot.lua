require 'gnuplot'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('PSNR / Loss plot comparison tool')
cmd:text()
cmd:text('Options:')
cmd:option('-display',      'false',    'Display the plot immediately')
cmd:option('-plotList',     '',         'Plots to compare (Delimiter is +)')
cmd:option('-xScale',       1e3,        'X axis scale')
cmd:option('-legend',       'rb',       'Position of legend')

local opt = cmd:parse(arg or {})
opt.display = opt.display == 'true'
opt.plotList = opt.plotList:split('+')

local lines = {}
local first, last = nil, nil

for i = 1, #opt.plotList do
    local function findMinMax(tb)
        local minKey, maxKey = math.huge, -math.huge    
        local minKeyValue, maxKeyValue
        for i = 1, #tb do
            if tb[i].key < minKey then
                minKey = tb[i].key
                minKeyValue = tb[i].value
            end
            if tb[i].key > maxKey then
                maxKey = tb[i].key
                maxKeyValue = tb[i].value
            end
        end
        return minKeyValue, maxKeyValue
    end

    local function toTensor(tb)
        local xAxis = {}
        local yAxis = {}
        for i = 1, #tb do
            table.insert(xAxis, tb[i].key)
            table.insert(yAxis, tb[i].value)
        end
        return torch.Tensor(xAxis), torch.Tensor(yAxis)
    end

    local tbl = torch.load(opt.plotList[i])
    for j = 1, #tbl do
        local xAxis, yAxis = toTensor(tbl[j])
        table.insert(lines, {i .. '_' .. j, xAxis:div(opt.xScale), yAxis, '-'})
    end
end

if not opt.display then
    local fig = gnuplot.pdffigure('plots.pdf')
end

gnuplot.plot(lines)
if opt.legend == 'rb' then
    gnuplot.movelegend('right', 'bottom')
else
    gnuplot.movelegend('right', 'top')
end
gnuplot.grid(true)
gnuplot.title('Plots')

local xlabel = 'Iterations'
if opt.xScale > 1 then
    xlabel = xlabel .. ' (*1e' .. math.log(opt.xScale, 10) .. ')'
end

gnuplot.xlabel(xlabel)

if not opt.display then
    gnuplot.plotflush(fig)
    gnuplot.closeall()
end