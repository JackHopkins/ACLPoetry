--file = torch.DiskFile('/home/rotmanmi/Data/GloVe/glove.twitter.27B.25d.txt','r')
--file = torch.DiskFile(opt.binfilename,'r')

--Reading Header

local encodingsize = -1
local ctr = 0
for line in io.lines(opt.glove) do
    if ctr == 0 then
        for i in string.gmatch(line, "%S+") do
            encodingsize = encodingsize + 1
        end
    end
    ctr = ctr + 1

end

words = ctr
size = encodingsize




local w2vvocab = {}
local v2wvocab = {}
local M = torch.FloatTensor(words,size)

--Reading Contents

i = 1
for line in io.lines(opt.glove) do
    xlua.progress(i,words)
    local vecrep = {}
    for i in string.gmatch(line, "%S+") do
        table.insert(vecrep, i)
    end
    str = vecrep[1]
    table.remove(vecrep,1)
	vecrep = torch.FloatTensor(vecrep)

	local norm = torch.norm(vecrep,2)
	if norm ~= 0 then vecrep:div(norm) end
	w2vvocab[str] = i
	v2wvocab[i] = str
	M[{{i},{}}] = vecrep
    i = i + 1
end


--Writing Files
GloVe = {}
GloVe.M = M
GloVe.w2vvocab = w2vvocab
GloVe.v2wvocab = v2wvocab
torch.save(opt.glove_output,GloVe)
print('Writing t7 File for future usage.')



return GloVe

