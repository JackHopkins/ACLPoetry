require 'util.misc'
-- WFST class using pretrained data
Carmel = {}

-- Create a Carmel
function Carmel:Create(carmelDir, wfstDir)


  local carmel = carmelDir
  local wfst = wfstDir
  local t = {}
local cache = {}
  t._et = {}


-- Executes Carmel process to retrieve all possible stress patterns for a line 
function t:get_stresses(input, stress_pattern) 

local cmd = ("echo '$foo' | "..carmel.."/carmel -sliOEQbk 1 "..wfst):gsub('$foo', input:upper())
local stress = os.capture(cmd)
if string.match(stress, "0 0 0 0 0 0 0 0 0 0") then return {} end

local current_patterns = {}
local pattern = ""
for i,v in pairs(split(stress, " ")) do 
             -- print(string.match(v, "%d+"))    
          if string.match(v, "%d+") then     
               
               local sub_pattern = string.sub(stress_pattern, 0, #pattern)
               local levenshtein = EditDistance(sub_pattern, pattern)
               

            if #pattern > 0 then
              if sub_pattern == pattern then
                current_patterns[pattern] = v
              elseif levenshtein < opt.stress_strictness then
                current_patterns[pattern] = v  
              end
            end
                pattern = ""
          else
                pattern = pattern .. string.gsub(string.gsub(v, "S%*", "/"), "S", "*")              
          end
    end

return current_patterns
end

function t:matches_stress(line, pattern, word_stack) 
   
  if not cache[line] then
   local stresses = self:get_stresses(line, long_pattern)
   
 --if not opt.stress_strictness == -1 then
    if count(stresses) == 0 then
      word_stack:pop(1)
      attempts = attempts + 1

      if attempts > 5 then 
       word_stack:pop(1) 
        attempts=0
      end
      return false, 0, 0
    end
--end

    local most_likely_stress, stress_prob = argmax(stresses)

    if #most_likely_stress > opt.syllables+1 then

       return false, 0, #most_likely_stress


    elseif math.abs(#most_likely_stress-opt.syllables) <= 1 then


         print("New Line: "..word_stack_to_string(word_stack))
         
         lines = lines + 1
       --  word_stack:pop(word_stack:getn())
    end

    cache[line] = {}
    cache[line][1] = stress_prob
    cache[line][2] = #most_likely_stress
 
    return true, stress_prob, #most_likely_stress

    else
     return true, cache[line][1], cache[line][2]
    end

end

return t
end