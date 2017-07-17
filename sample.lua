--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'carmel'

require 'util.OneHot'
require 'util.misc'
require 'util.stack'
require 'util.wfst'
local WordSplitLMMinibatchLoader = require 'util.WordSplitLMMinibatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temp_mean',0.5,'temperature of sampling')
cmd:option('-temp_dev', 0, 'how much the temperature varies')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',1,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',0,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-stress_strictness',0,'numbers of allowed rhythmic mistakes. -1 is least strict.')
cmd:option('-name','results', 'name of result file')
cmd:option('-syllables',10,'number of syllables per line')
cmd:option('-pattern', "*/", 'stress pattern')
cmd:option('-alliteration', 3, 'alliteration coefficient')
cmd:option('-branchingFactor', 10, 'How many solutions we keep in the beam.')
cmd:option('-theme', "cold", "A theme word to guide the poetry sampling.")
cmd:option('-form', 0.5, "The weighting of form with respect to content")
cmd:option('-carmel', 0, "Whether to use the Carmel library")
cmd:option('-carmel_dir', '/Users/jack/Documents/workspace/Poebot/graehl/carmel/bin/macosx')
cmd:option('-wfst', '/Users/jack/Documents/workspace/Poebot/torch-rnn/torch-rnn/wfst005.full.txt')
cmd:option('-output_dir', '/Users/jack/Documents/workspace/Poebot/lstm/src/data/')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
current_state = {}
prediction = {}
word_stack = Stack:Create()
lines = 0
--temperature = opt.temperature
drift = 0
theme = {}


if opt.carmel == 0 then
pronunciation_model = WFST:Create()
pronunciation_model:load(opt.wfst)
else
pronunciation_model = Carmel:Create(opt.carmel, opt.wfst)
end


long_pattern = ""
   for i=1, 20 do
        long_pattern=long_pattern..opt.pattern
    end

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end



function create_theme_graph(theme_word)
for i=1,string.len(theme_word)-1 do
local head = string.sub(theme_word, i, i+1) 
local tail = string.sub(theme_word, i+1)

if not theme[tail] then theme[tail] = {} end
if not theme[tail][head] then theme[tail][head] = 0 end

if theme[tail][head] then
theme[tail][head] = 1
else

theme[tail][head] = theme[tail][head] + 1

end
end
end



function word_stack_to_string(word_stack)
  local words = ""
  if word_stack:getn() == 0 then
    return words
  end
  for i,v in pairs(word_stack._et) do 
    words = words..v

  end
  return words
end

function boost_theme(probabilities, incomplete_word) 
local theme_word = opt.theme
if string.len(incomplete_word) > string.len(theme_word) then return probabilities end

if incomplete_word == string.sub(theme_word, 1, string.len(incomplete_word)) then

--  print(theme[incomplete_word])
  if theme[incomplete_word] then
  for key,value in pairs(theme[incomplete_word]) do
      probabilities[vocab[key]] = 1--probabilities[vocab[key]] * value * theme_constant
  end
end
end
  return probabilities
end

function alliteration(probabilities, word_stack, incomplete_word)
  local letter_set = {}
  for key, value in word_stack:asList() do
    local char = string.sub(value, 1, 1)
    letter_set[char] = true
  end

  for key, value in pairs(letter_set) do
    local char = key
  probabilities[vocab[char]] = probabilities[vocab[char]] * opt.alliteration
  end
         -- renormalize so probs sum to one
  return probabilities:div(torch.sum(probabilities))
end

function sample_new_word(temperature, word_stack)

local word = ""
local total_probability = 1

    while (true) do
     -- log probabilities from the previous timestep
    if opt.sample == 0 then
        -- use argmax
        local _, prev_char_ = prediction:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        prediction:div(temperature) -- scale by temperature

        local probs = boost_theme(torch.exp(prediction):squeeze(), word)
        probs = alliteration(probs, word_stack, word)
        if torch.sum(probs) == 0 then goto continue end
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
        total_probability = total_probability*probs[prev_char[1]]
    end

    -- forward the rnn for next character
    local lst = protos.rnn:forward{prev_char, unpack(current_state)}
    current_state = {}

    for i=1,state_size do table.insert(current_state, lst[i]) end

    prediction = lst[#lst] -- last element holds the log probabilities

   
    local char = ivocab[prev_char[1]]

  -- Remove sampled break-lines and convert them to spaces
   char = string.gsub(char, "\n", " ")
   if string.find(char, "[%p%d]") then
    goto continue
   end

    word = word..char
    begin = true
    
    
    --If a space has been found, 
    if char == " " then 
        --and the word solely consists of that character
        if #word == 1 then
          goto continue
          --or the word is longer than 1
        elseif #word > 1 then
        break
        end
    end 
    ::continue::
    end

    return word, totalProbability

end




function seed(seed_text) 
gprint(seed_text)
local seedlist
    if(checkpoint.opt.wordlevel==1) then
       local words=WordSplitLMMinibatchLoader.preprocess(seed_text)
       seedlist = words:gmatch("([^%s]+)")
    else
        seedlist = seed_text:gmatch'.'
    
    end

    for c in seedlist do
    
        local idx = vocab[c]
        if idx == nil then idx = vocab[unknownword] end
        prev_char = torch.Tensor{vocab[c]}
        io.write(ivocab[prev_char[1]])
        if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
        if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
        local lst = protos.rnn:forward{prev_char, unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        prediction = lst[#lst] -- last element holds the log probabilities
    end

end

function argmax(stresses)
  local best_prob = 0
  local best_stress = ""
  for i,v in pairs(stresses) do 
    if tonumber(v) > best_prob then
      best_prob = tonumber(v)
      best_stress = i
    end

  end
  return best_stress, best_prob
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
        cutorch.manualSeed(opt.seed)
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
       -- torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
vocab = checkpoint.vocab
ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
--local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end
state_size = #current_state

-- do a few seeded timesteps
local seed_text = opt.primetext
local unknownword = "<unk>"


if string.len(seed_text) > 0 then
    gprint('seeding with ' .. seed_text)
    gprint('--------------------------')
    
    seed(seed_text, current_state)
else
    -- fill with uniform probabilities over characters (? hmm)
    gprint('missing seed text, using uniform probability over first character')
    gprint('--------------------------')
    prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
    if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
end

function get_lines_sorted_by_score(tbl, sort_function)
  local keys = {}
  for key, _ in pairs(tbl) do
    table.insert(keys, key)
  end

  table.sort(keys, function(a, b) return sort_function(tbl[a], tbl[b]) end)

  return keys
end
function objective(a) 
  return ((a.score*opt.form)+(a.probability*(1-opt.form)))*a.stack:getn()
end
function compareLines(a, b) 
  --if a.syllables >= opt.syllables then return 1 end
  --if b.syllables >= opt.syllables then return 0 end
  return objective(a) > objective(b)
end
function prune(candidates, max) 

  local new_candidates = {}
  local sortedKeys = get_lines_sorted_by_score(candidates, compareLines)
  for k,v in pairs(sortedKeys) do
    print(objective(candidates[sortedKeys[k]]).." "..candidates[sortedKeys[k]].line)
    if k <= max then
      new_candidates[sortedKeys[k]] = candidates[sortedKeys[k]]
    end
  end
  print(max)
  print(tablelength(candidates))
  print(tablelength(new_candidates))

  --for i = 1, max do 
  --  new_candidates[sortedKeys[i]] = candidates[sortedKeys[i]]
  --end
  

  return new_candidates
end


function choose_temperature()
  local temperature = 0
    while temperature <= 0 or temperature > 1 do
        temperature = opt.temp_mean + (math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * opt.temp_dev)
    end
    if temperature == nil then return 1 end
    return temperature
  end
-- For every candidate, sample another word 
function beam_search(candidates, stress_implementation) 
  local candidateNum = 0
    for key, value in pairs(candidates) do
  
    --By varying the temperature each line is samples at, we can better traverse the search space
    local temperature = choose_temperature()
  --  print(temperature)

    local candidate = sample_line(value.stack, value.probabilities, candidates, temperature, stress_implementation)
    

    if candidate ~= nil then
      candidates[candidate.line] = candidate
    end

    candidateNum = candidateNum + 1
    end
  return candidates
end


function seed_candidates(candidates)
  local start_line = nil

  while(start_line == nil) do 
    start_line = sample_line(Stack:Create(), Stack:Create(), candidates, 1)--choose_temperature())
  end

  while (tablelength(candidates) < opt.branchingFactor) do
    local temp = choose_temperature()
    start_line = sample_line(Stack:Create(), Stack:Create(), candidates, temp)
  
    if start_line ~= nil then
      candidates[start_line.line] = start_line
      --temperatures[start_line.]
    end

  end

end

function sample_line(word_stack, probability_stack, candidates, temperature)
  local lineProbability = 1
  local words = word_stack_to_string(word_stack)

  -- Condition the RNN to the words already in the line
  seed(words)

  local word_stack_n = word_stack:clone()
  local word_sample, word_probability = sample_new_word(temperature, word_stack_n)

 
  local probability_stack_n = probability_stack:clone()

  word_stack_n:push(word_sample)
  probability_stack_n:push(word_probability)

words = sanitise(word_stack_to_string(word_stack_n))
--print("words: "..words)
local matches = true
local score = 0
local syllables = 0

-- If we care about the stresses...
if opt.stress_strictness ~= -1 then
    matches, score, syllables = pronunciation_model:matches_stress(words, opt.pattern, word_stack)
end

  -- If the line conforms to the stress pattern, print it. If not, continue
  if matches then
    
    local probability = 1

    for k,v in probability_stack_n:asList() do
      probability = probability * v
   
    end
    probability = probability
    return {stack = word_stack_n, score = score, line = word_stack_to_string(word_stack_n), probabilities = probability_stack_n, probability = probability, syllables = syllables}
  end

  return nil
end


-- start sampling/argmaxing
local line = opt.primetext
seed(line)
attempts = 0



::start::
create_theme_graph(opt.theme)

local candidates = {}
local temperatures = {}
seed_candidates(candidates)


while(true) do
  candidates = beam_search(candidates, carmel)

  if tablelength(candidates) >= opt.branchingFactor then 
  candidates = prune(candidates, opt.branchingFactor)
  end

  for k,v in pairs(candidates) do 
    if (v.syllables >= opt.syllables) then
      local output = assert(io.open(opt.output_dir..opt.name.."."..opt.stress_strictness.."."..opt.temp_mean.."."..opt.temp_dev..".csv", 'a'))
         output:write(k..","..v.score..","..v.probability.."\n" )
         io.close(output)
      goto start
    end
  end
end



io.write('\n') io.flush()
