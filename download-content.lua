local cURL = require 'cURL'
local paths = require 'paths'

require 'tar.lua'

-- Create the directories if needed
if not paths.dirp('dataset') then paths.mkdir('dataset') end

-- Url, location, destination
local fileList = {
	{'https://www.adrianbulat.com/downloads/datasets/AFLW2000.tar', 'dataset/AFLW2000.tar', 'dataset/'},
	{'https://www.adrianbulat.com/downloads/datasets/AFLWPIFA-val.tar', 'dataset/AFLWPIFA-val.tar', 'dataset/'}
}

local fileListCopy = {}
local m = cURL.multi()

for i = 1, #fileList do
	fileListCopy[i] = {fileList[i][2], fileList[i][3]}

	-- Open files
	fileList[i][2] = io.open(fileList[i][2], "w+b")

	-- Add the url handles
	fileList[i][1] = cURL.easy{url = fileList[i][1], writefunction = fileList[i][2]}
	m:add_handle(fileList[i][1])
end

print("Downloading files, please wait...")
-- Based on https://github.com/Lua-cURL/Lua-cURLv3/blob/master/examples/cURLv3/multi2.lua
local remain = #fileList
while remain > 0 do
	local last = m:perform()
	if last < remain then
		while true do
			local e, ok, err = m:info_read(true)
			if e == 0 then break end -- no more finished tasks
			if ok then
				print(e:getinfo_effective_url(), '-', '\027[00;92mOK\027[00m')
			else
				print(e:getinfo_effective_url(), '-', '\027[00;91mFail\027[00m')
			end
			e:close()
		end
	end 
	remain = last

	m:wait() 
end	

-- Untar files and delete them afterwards
print('Uncompressing files...')
for i = 1, #fileListCopy do
    untar(fileListCopy[i][1], fileListCopy[i][2], true, (function() print(paths.basename(fileListCopy[i][1])..' uncompressed succesfully') end))
end

