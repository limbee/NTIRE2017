function ret = getKey(hFig)
    ind = 777;
    while (ind ~= 1)
        ind = waitforbuttonpress;
    end
    ret = hFig.CurrentCharacter;
end