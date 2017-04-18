require 'image'

local M = {}

function M.Compose(transforms)
    return function(sample)
        for _, transform in ipairs(transforms) do
            sample = transform(sample)
        end
        return sample
    end
end

function M.HorizontalFlip()
    return function(sample)
        if torch.uniform() < 0.5 then
            sample.input = image.hflip(sample.input)
            sample.target = image.hflip(sample.target)
        end
        return sample
    end
end

function M.Rotation()
    return function(sample)
        local theta = torch.random(0,3)
        sample.input = image.rotate(sample.input, theta * math.pi/2)
        sample.target = image.rotate(sample.target, theta * math.pi/2)
        return sample
    end
end

local function blend(img1, img2, alpha)
    return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
    dst:resizeAs(img)
    dst[1]:zero()
    dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
    dst[2]:copy(dst[1])
    dst[3]:copy(dst[1])
    return dst
end

function M.Saturation(var)
    local gs

    return function(sample)
        gs = gs or sample.input.new()
        grayscale(gs, sample.input)

        local alpha = 1.0 + torch.uniform(-var, var)
        blend(sample.input, gs, alpha)

        gs = sample.target.new()
        grayscale(gs, sample.target)
        blend(sample.target, gs, alpha)

        return sample
    end
end

function M.Brightness(var)
    local gs

    return function(sample)
        gs = gs or sample.input.new()
        gs:resizeAs(sample.input):zero()

        local alpha = 1.0 + torch.uniform(-var, var)
        blend(sample.input, gs, alpha)

        gs = sample.target.new()
        gs:resizeAs(sample.target):zero()
        blend(sample.target, gs, alpha)

        return sample
    end
end

function M.Contrast(var)
    local gs

    return function(sample)
        gs = gs or sample.input.new()
        grayscale(gs, sample.input)
        gs:fill(gs[1]:mean())

        local alpha = 1.0 + torch.uniform(-var, var)
        blend(sample.input, gs, alpha)

        gs = sample.target.new()
        grayscale(gs, sample.target)
        gs:fill(gs[1]:mean())
        blend(sample.target, gs, alpha)

        return sample
    end
end

function M.RandomOrder(ts)
    return function(sample)
        local order = torch.randperm(#ts)
        for i = 1,#ts do
            sample = ts[order[i]](sample)
        end
        return sample
    end
end

function M.ColorJitter(opt)
    local brightness = opt.brightness
    local contrast = opt.contrast
    local saturation = opt.saturation

    local ts = {}
    if brightness ~= 0 then
        table.insert(ts, M.Brightness(brightness))
    end
    if contrast ~= 0 then
        table.insert(ts, M.Contrast(contrast))
    end
    if saturation ~= 0 then
        table.insert(ts, M.Saturation(saturation))
    end

    if #ts == 0 then
        return function(sample) return sample end
    end

    return M.RandomOrder(ts)
end

return M
