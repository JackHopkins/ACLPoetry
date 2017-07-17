
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
require 'util.phone_lookup'

local WordSplitLMMinibatchLoader = require 'util.WordSplitLMMinibatchLoader'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-wordmodel','model checkpoint to use for sampling')
cmd:argument('-phonemodel','model checkpoint to use for sampling')
cmd:argument('-cmu','CMU dictionary')
-- optional parameters
cmd:option('-seed',99,'random number generator\'s seed')
cmd:option('-formatwords',0,'Insert appropriate spaces in text')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-phoneticTemperature',0.4,'temperature for sampling the phonetic model')
cmd:option('-wordTemperature',0.0,'temperature for word sampling restricton. 1=argmax 0=none')

cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()


function translateWords(words)

 
  return words

end

function table.contains(table, element)
  for _, value in pairs(table) do
    if value == element then
      return true
    end
  end
  return false
end

function hasValue (tab, val)
    for index, value in ipairs (tab) do
        if value == val then
            return true
        end
    end

    return false
end

function hasKey (tab, key)
    for index, value in ipairs (tab) do
        if index == key then
            return true
        end
    end

    return false
end


-- (OPEN-PARENTHESES  OW1 P AH0 N P ER0 EH1 N TH AH0 S IY2 Z
function loadCmuLookup(filename)
  vocab = {}
  ivocab = {}
   
  -- Set at 2 to account for the space and newline characters
  index = 3;
 

  

  for l in io.lines(filename) do
    local w, p = l:match '(.*)  (.+)'

    vocab[w] = p.gsub(p,"[012]","");
    --print(  vocab[w] )
    ivocab[p.gsub(p,"[012]","")] = w;

    phonelist = p.gsub(p,"[012]",""):gmatch'[^%s]+'
    print(w .. "  " .. p)
  
  

 -- for phone in phonelist do

   -- if not phoneIVocab[phone] then
       -- phoneVocab[index] = phone
       -- phoneIVocab[phone] = index
       -- index = index + 1
  --  end 
   
  --end 

  --  phoneVocab[
  end
--print(phoneIVocab)

  return { vocab = vocab, ivocab=ivocab}
end

function ParseCSVLine (line,sep) 
    local res = {}
    local pos = 1
    sep = sep or ','
    while true do 
        local c = string.sub(line,pos,pos)
        if (c == "") then break end
        if (c == '"') then
            -- quoted value (ignore separator within)
            local txt = ""
            repeat
                local startp,endp = string.find(line,'^%b""',pos)
                txt = txt..string.sub(line,startp+1,endp-1)
                pos = endp + 1
                c = string.sub(line,pos,pos) 
                if (c == '"') then txt = txt..'"' end 
                -- check first char AFTER quoted string, if it is another
                -- quoted string without separator, then append it
                -- this is the way to "escape" the quote char in a quote. example:
                --   value1,"blub""blip""boing",value3  will result in blub"blip"boing  for the middle
            until (c ~= '"')
            
            for k, v in txt.gmatch(s, "(.*)\t(.+)") do
           table.insert(k, v)
           print(k .. " - " .. v)
            end
            
            
            assert(c == sep or c == "")
            pos = pos + 1
        else    
            -- no quotes used, just look for the first separator
            local startp,endp = string.find(line,sep,pos)
            if (startp) then 
                table.insert(res,string.sub(line,pos,startp-1))
                pos = endp + 1
            else
                -- no separator found -> use rest of string and terminate
                table.insert(res,string.sub(line,pos))
                break
            end 
        end
    end
    return res
end




-- parse input params
opt = cmd:parse(arg)

cmu = loadCmuLookup(opt.cmu)



-- gated print: simple utility function wrapping a print
function gprint(str)
  if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then gprint('package cunn not found!') end
  if not ok2 then gprint('package cutorch not found!') end
  if ok and ok2 then
    gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
    gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    if opt.seed ~= 99 then cutorch.manualSeed(opt.seed) end
  else
    gprint('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
  local ok, cunn = pcall(require, 'clnn')
  local ok2, cutorch = pcall(require, 'cltorch')
  if not ok then print('package clnn not found!') end
  if not ok2 then print('package cltorch not found!') end
  if ok and ok2 then
    gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
    gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
    cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    torch.manualSeed(opt.seed)
  else
    gprint('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.wordmodel, 'mode') then
  gprint('Error: File ' .. opt.wordmodel .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end

phoneCheckpoint = torch.load(opt.phonemodel)

phoneEncoding = phoneCheckpoint.vocab
phoneIEncoding = {}


phoneVocab = {} -- phoneCheckpoint.vocab
phoneIVocab = {
     
     ['\n']=  1,
     [' ']=   2,

     ["AA"]=  3,
     ["AE"]=  4,
     ["AH"]=  5,
     ["AX"]=  5,
     ["AO"]=  6,
     ["AW"]=  7,
     ["AY"]=  8,
     ["B"]=  9,
     ["CH"]=  10,
     ["D"]=  11,
     ["DH"]=  12,
     ["EH"]=  13,
     ["ER"]=  14,
     ["EY"]=  15,
     ["F"]=   16,
     ["G"]=   17,
     ["HH"]=  18,
     ["IH"]=  19,
     ["IY"]=  20,
     ["JH"]=  21,
     ["K"]=  22,
     ["L"]=  23,
     ["M"]=  24,
     ["N"]=  25,
     ["NG"]=  26,
     ["OW"]=  27,
     ["OY"]=  28,
     ["P"]=  29,
     ["R"]=  30,
     ["S"]=  31,
     ["SH"]=  32,
     ["T"]=  33,
     ["TH"]=  34,
     ["UH"]=  35,
     ["UW"]=  36,
     ["V"]=  37,
     ["W"]=  38,
     ["Y"]=  39,
     ["Z"]=  40,
     ["ZH"]=  41,
     [';']= 42
     }

wordCheckpoint = torch.load(opt.wordmodel)


wordProtos = wordCheckpoint.protos
phoneProtos = phoneCheckpoint.protos

wordProtos.rnn:evaluate() -- put in eval mode so that dropout works properly
phoneProtos.rnn:evaluate()


-- initialize the vocabulary (and its inverted version)
local wordVocab = wordCheckpoint.vocab
local wordIVocab = {}

-- Used for debugging 
--local phoneVocab = phoneCheckpoint.vocab
--local phoneIVocab = {}

--for c,i in pairs(pVocab) do pIVocab[i] = c end
for c,i in pairs(wordVocab) 
  do wordIVocab[i] = c
 end

for c,i in pairs(phoneVocab) do phoneIVocab[i] = c end
for c,i in pairs(phoneEncoding) do phoneIEncoding[i] = c end

-- initialize the rnn state to all zeros
gprint('creating a word ' .. wordCheckpoint.opt.model .. '...')
gprint('creating a phone ' .. phoneCheckpoint.opt.model .. '...')


-- set word current state
local word_current_state
word_current_state = {}
for L = 1,wordCheckpoint.opt.num_layers do
  -- c and h for all layers
  local h_init = torch.zeros(1, wordCheckpoint.opt.rnn_size):double()
 -- if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
 -- if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
  table.insert(word_current_state, h_init:clone())
  if wordCheckpoint.opt.model == 'lstm' then
    table.insert(word_current_state, h_init:clone())
  end
end
word_state_size = #word_current_state


-- set phone current state
local phone_current_state
phone_current_state = {}
for L = 1,phoneCheckpoint.opt.num_layers do
  -- c and h for all layers
  local h_init = torch.zeros(1, phoneCheckpoint.opt.rnn_size):double()
  if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
  if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
  table.insert(phone_current_state, h_init:clone())
  if phoneCheckpoint.opt.model == 'lstm' then
    table.insert(phone_current_state, h_init:clone())
  end
end
phone_state_size = #phone_current_state 


gprint("Word RNN Size: "..wordCheckpoint.opt.rnn_size)
gprint("Word RNN N Layers: "..wordCheckpoint.opt.num_layers)
gprint("Word Dropout: "..wordCheckpoint.opt.dropout)

gprint("Phone RNN Size: "..phoneCheckpoint.opt.rnn_size)
gprint("Phone RNN N Layers: "..phoneCheckpoint.opt.num_layers)
gprint("Phone Dropout: "..phoneCheckpoint.opt.dropout)

gprint("NOTE: Seeding only seeds the phonetic RNN currently.")

-- do a few seeded timesteps
local seed_text = opt.primetext
local unknownword = "<unk>"

function seed (seed_text)

if not seed_text == nil then
seed_text = strupper(seed_text)
end

-- Fill with uniform probabilities

wordPrediction = torch.Tensor(1, #wordIVocab):fill(1)/(#wordIVocab)
if opt.gpuid >= 0 and opt.opencl == 0 then wordPrediction = wordPrediction:cuda() end
if opt.gpuid >= 0 and opt.opencl == 1 then wordPrediction = wordPrediction:cl() end


  if string.len(seed_text) > 0 then
    gprint('seeding with ' .. translateWords(seed_text))
    gprint('--------------------------')


   -- local seedlist
    
    
    --if(checkpoint.opt.wordlevel==1) then
    
    local words=WordSplitLMMinibatchLoader.preprocess(seed_text)
      --print(words)
     local seedlist = words:gmatch("([^%s]+)")
    
      
   -- else
   --   seedlist = seed_text:gmatch'.'
   -- end

    local seedtext = ""
    
    
    for c in seedlist do
      phonelist = cmu.vocab[c]:gmatch'[^%s]+' 
      

      -- print(phoneVocab)
      for p in phonelist do 
     
   
      local idx = phoneIVocab[p]
     -- print(phoneIVocab[p])
      
      if idx == nil then
        idx = phoneIVocab[unknownphone] 
        prev_char = torch.Tensor{phoneIVocab[unknownword]}
      else
       prev_char = torch.Tensor{phoneIVocab[p]}
      end
     

      seedtext=seedtext..phoneVocab[prev_char[1]]
      
      if(phoneCheckpoint.opt.wordlevel==1) then seedtext=seedtext.." " end
      if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
      if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end

      local lst = phoneProtos.rnn:forward{prev_char, unpack(phone_current_state)}
      -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
      phone_current_state = {}
      for i=1,phone_state_size do table.insert(phone_current_state, lst[i]) end
      
      phonePrediction = lst[#lst] -- last element holds the log probabilities
    
    end
    end
    
    --io.write(translateWords(seedtext))
  else
    -- fill with uniform probabilities over characters (? hmm)
    gprint('missing seed text, using uniform probability over first character')
    gprint('--------------------------')
    gprint(phoneEncoding)
    phonePrediction = torch.Tensor(1, #phoneIEncoding):fill(1)/(#phoneIEncoding)
    if opt.gpuid >= 0 and opt.opencl == 0 then phonePrediction = phonePrediction:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then phonePrediction = phonePrediction:cl() end
  end

end

--seed (seed_text)

function samplePhones()
  
  -- start sampling/argmaxing
  local phones=""
  local words=""

  local length = opt.length

  -- where we store the word-level checkpoint to revert the rnn
  local phone_current_state_checkpoint = {}
  for i=1,phone_state_size do table.insert(phone_current_state_checkpoint, phone_current_state[i]) end

  local phone_current_state_checkpoint_size = #phone_current_state_checkpoint

  -- how many phones are in the word currently being sampled
  local number_phones_in_word = 0

  local i = 1;
  --for i=1, length do
  while (true) do
    -- log probabilities from the previous timestep
    if opt.sample == 0 then
      -- use argmax
      local _, prev_char_ = phonePrediction:max(2)
      prev_char = prev_char_:resize(1)
    else

     
      -- use sampling
      phonePrediction:div(opt.phoneticTemperature) -- scale by temperature
      local probs = torch.exp(phonePrediction):squeeze()


      probs:div(torch.sum(probs)) -- renormalize so probs sum to one
      prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
      --print(embeddingToPhone(phoneIEncoding[prev_char[1]]))

     

    end


    -- forward the rnn for next character
    phoneLst = phoneProtos.rnn:forward{prev_char, unpack(phone_current_state)}
    phone_current_state = {}
    for i=1,phone_state_size do table.insert(phone_current_state, phoneLst[i]) end
    phonePrediction = phoneLst[#phoneLst] -- last element holds the log probabilities


    local phone = embeddingToPhone(phoneIEncoding[prev_char[1]])

    -- increment number of phones
    number_phones_in_word = number_phones_in_word + 1

  --lookup index 0 = space
  if string.len(phones) == 0 then
    phones = phone 
  else 
    phones = phones..' '..phone
  end

if phoneIEncoding[prev_char[1]] == ';' then
  words = words..'\n'
end

 -- if prev_char is a \n = new poem
  if prev_char[1] == 1 then
    print(words.." ")
    words = "";
    phones = "";
  end

   -- if prev_char is a space = new word
  if prev_char[1] == 2 then

    local trimmedPhones = string.sub(phones, 0, -3)
    local word = ivocab[trimmedPhones]

    --print(word)
    --print(trimmedPhones)
   
  -- decrement number of phones, as the 'phone' is actually a space
    number_phones_in_word = number_phones_in_word - 1

  -- if the word exists and is plausible
      if evaluateWord(word) then
        --advanceWordRNN(word)
        words = words..' '..word

         -- populate checkpoint
         phone_current_state_checkpoint = {}
         for i=1,phone_state_size do table.insert(phone_current_state_checkpoint, phoneLst[i]) end
         phone_current_state_checkpoint_size = #phone_current_state_checkpoint

      else 

      --retreat 
   --   if (word) then
   --   print('retreating -'..word)
   -- end

      phone_current_state = {}
      for i=1,phone_current_state_checkpoint_size do table.insert(phone_current_state, phone_current_state_checkpoint[i]) end
      phone_state_size = #phone_current_state

      -- add another thing to sample
      length = length+1
      --print(i.."/"..length)
      end


  number_phones_in_word = 0
  phones = ""
  i = i+1

  if (i == length) then break; end

  end 
end

  io.write('\n') io.flush()
end

function advanceWordRNN(prev_word) 

   local lst = wordProtos.rnn:forward{prev_word, unpack(word_current_state)}
    word_current_state = {}
    for i=1,word_state_size do table.insert(word_current_state, lst[i]) end
    wordPrediction = lst[#lst] -- last element holds the log probabilities

end 

function evaluateWord(word)

if not word then 
  return false
end

-- if 0 then no semantic classification is to be performed
if opt.wordTemperature == 0 then
  return true
end

--wordPrediction:div(opt.temperature) -- scale by temperature

local probs = torch.exp(wordPrediction):squeeze()

 -- renormalize so probs sum to one
probs:div(torch.sum(probs))

-- local prev_word = torch.Tensor(probs:narrow(1,2,1):float())--torch.multinomial(probs:float(), 1):resize(1):float()


--We need to sanitise our CMU words so they can be found in our training vocabulary
local sanitisedWord = string.gsub(word, "%(%d%)", ""):lower()  

-- If this CMU word is not found in our training vocab, we must try again
if not wordVocab[sanitisedWord] then
  print(word .. " is not recognised.")
return false
end

-- If word temperature is 0 then there is no semantic filtering
--if opt.wordTemperature == 0 then
--return true
--end

local wordIndex = wordVocab[sanitisedWord] 

local _, argmaxword_ = wordPrediction:max(2)
     local argmaxword = argmaxword_:resize(1)

local maxProb = probs[argmaxword[1]]
local wordProb= probs[wordIndex]

-- Whats the lowest probability our temperature will allow?
local minProbability = maxProb*opt.wordTemperature


if (wordProb < minProbability) then
--  print("Argmax "..wordIVocab[argmaxword[1]].." - "..maxProb)
--  print("Select "..wordIVocab[wordIndex].." - "..wordProb)
--print(sanitisedWord..": "..wordProb.." < "..minProbability)
--print("NO "..sanitisedWord..": "..wordProb)
return false
end

--print("YES "..sanitisedWord..": "..wordProb)

local prev_word = torch.Tensor{wordIndex}-- torch.multinomial(probs:float(), 1):resize(1):float()
--print(torch.Tensor{wordIVocab[wordIndex]})
--print(torch.multinomial(probs:float(), 1):resize(1):float())
--print("Argmax "..wordIVocab[argmaxword[1]].." - "..probs[argmaxword[1]])
--print("Select "..wordIVocab[wordIndex].." - "..probs[wordIndex])


advanceWordRNN(prev_word)
--print(wordPrediction)
--print(torch.sum(probs))


return true
end

seed("")
samplePhones()
