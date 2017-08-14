torch.setdefaulttensortype('torch.FloatTensor')
require 'util.misc'

local GloVe = {}
if not file_exists(opt.glove_output) then
	print("Loading word embeddings from GloVe. \nThis is only done when first running the program.")
	GloVe = require('bintot7')
else
	print('Reading GloVe data.')
	GloVe = torch.load(opt.glove_output)
	print('Done reading GloVe data.')
end


GloVe.distance = function (self,vec,k)
	local k = k or 1	
	--self.zeros = self.zeros or torch.zeros(self.M:size(1));
	local norm = vec:norm(2)
	vec:div(norm)
	local distances = torch.mv(self.M ,vec)
	distances , oldindex = torch.sort(distances,1,true)
	local returnwords = {}
	local returndistances = {}
	for i = 1,k do
		table.insert(returnwords, self.v2wvocab[oldindex[i]])
		table.insert(returndistances, distances[i])
	end
	return {returndistances, returnwords}
end

GloVe.word2vec = function (self,word,throwerror)
   local throwerror = throwerror or false
   local ind = self.w2vvocab[word]
   if throwerror then
		assert(ind ~= nil, 'Word does not exist in the dictionary!')
   end
	if ind == nil then
		ind = self.w2vvocab['UNK']
	end
   return self.M[ind]
end

return GloVe