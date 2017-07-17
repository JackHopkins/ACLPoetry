encoding = {
		[" "] = ' ',
		['\n'] = '\n',
		[';'] = ';',
		["AA"] =  '0',
		["AE"] =  '1',
		["AX"] =  '2',
		["AH"] =  '2',
		["AO"] =  '3',
		["AW"] =  '4',
		["AY"] =  '5',
		["B"] =  '6',
		["CH"] =  '7',
		["D"] =  '8',
		["DH"] =  '9',
		["EH"] =  'a',
		["ER"] =  'b',
		["EY"] =  'c',
		["F"] =  'd',
		["G"] =  'e',
		["HH"] =  'f',
		["IH"] =  'g',
		["IY"] =  'h',
		["JH"] =  'i',
		["K"] =  'j',
		["L"] =  'k',
		["M"] =  'l',
		["N"] =  'm',
		["NG"] =  'n',
		["OW"] =  'o',
		["OY"] =  'p',
		["P"] =  'q',
		["R"] =  'r',
		["S"] =  's',
		["SH"] =  't',
		["T"] =  'u',
		["TH"] =  'v',
		["UH"] =  'w',
		["UW"] =  'x',
		["V"] =  'y',
		["W"] =  'z',
		["Y"] =  'A',
		["Z"] =  'B',	
		["ZH"] =  'C'
}
iencoding = {}

for c,i in pairs(encoding) do iencoding[i] = c end

phones = {"AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"}
char_embedding = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}
function encodePhones(phone) 
	return encoding[phone]
end

function decodePhones(index)
	return phones[index]
end

function embeddingToPhone(character) 
	if iencoding[character] == "AX" then return "AH" end
	return iencoding[character]
end 