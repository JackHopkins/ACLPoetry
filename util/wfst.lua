require 'util.misc'
-- WFST class using pretrained data
WFST = {}

-- Create a WFST
function WFST:Create()

  -- stack table
  local t = {}
  -- entry table
  t._et = {}



function t:parse(line)
  local bits = split(line, " ")

  local key = bits[2]:sub(3) 
  local word = bits[3]:sub(2, -2)
local probability = bits[5]:sub(1,-3)
  return key, word, probability
end

  function t:forward(line, index, remaining_stress, probability, length)
  
    if line[index] == nil then return probability, length end

    local word = line[index]:sub(2,-2)
    local stress = self:get_stress(word, remaining_stress)
    local n_probability = 0
    local max_length = 0
    for len,v in pairs(stress) do 
     
      local clipped_stress = remaining_stress:sub(len+1)

      -- return the probability of one path, and the length of that path
     local local_prob, local_length = self:forward(line, index+1, clipped_stress, probability*v, len+length)
  
     n_probability = n_probability + local_prob
     if local_length > max_length then max_length = local_length end
    end

    return n_probability, max_length

  end

  function t:get_line_stress_probability(line, stress_pattern)


    local parts = split(line:upper(), " ")
    
    return self:forward(parts, 1, stress_pattern, 1, 0)
  end

  --function t:probability_word_in_pattern(word, stress_pattern) 

  --end

function t:matches_stress(line, pattern, word_stack)  

  local long_stress = pattern
for i = 0, 10 do
  long_stress = long_stress..pattern
end 

    local probability, max_length = self:get_line_stress_probability(line, long_stress)
  

  if max_length ~= 0 then
    return true, probability, max_length
  else
    return false, probability, 0
  end
end



  function t:get_stress(word, stress_pattern)
   --local millis = os.clock()*10
  
    if self._et[word] == nil then return {} end

    stress_pattern = stress_pattern:gsub("*", "0"):gsub("/", "1")

    local probability = {}
    
    for k, v in pairs(self._et[word]) do
      local sub_pattern = stress_pattern
      local actual = v[1]
      
     if #stress_pattern > #actual then
        sub_pattern = string.sub(stress_pattern, 0, #actual)
      else
        actual = string.sub(actual, 0, #stress_pattern)
      end
      
   --   print(stress_pattern.." - "..sub_pattern.." - "..v[2].." - "..actual)
      if EditDistance(sub_pattern, actual) == 0 then
        if probability[#actual] == nil then 
          probability[#actual] = v[2] 
        else
        probability[#actual] = probability[#actual] + v[2]
        end
      end

    end

    return probability
  end




  function t:load(file) 
  local f = io.open(file, "rb")
  local content = ""
  local length = 0

  local first = true
  while true do
    local line = f:read("*l")
    if first ~= true then
    if line == nil then break end
    key, word, probability = self:parse(line)

    if t._et[word] == nil then
      t._et[word] = {}
      t._et[word][1] = {key, probability}
     --print(t._et[word][key] = probability 
  
       --table.insert(t[word], key) 
    else
      t._et[word][#t._et[word]+1] = { key, probability }
    end


    if line == nil then break end
    end
  first = false
  
  end

  end


  function t:clone() -- deep-copy a table
    
    local target = WFST:Create()

    for k, v in pairs(self._et) do
        if type(v) == "table" then
            target:push(clone(v))
        else
            target:push(v)
        end
    end
   -- setmetatable(target, meta)

    return target
end


  -- get entries
  function t:getn()
    return #self._et
  end
  function t:get(index)
    return self._et
  end

 
  function t:asList() 
    return pairs(self._et)
  end
  -- list values
  function t:list()
    for i,v in pairs(self._et) do
      print(i, v)
    end
  end
  return t
end
