local function parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Binary Face alignment demo script')
	cmd:text('Please visit https://www.adrianbulat.com for additional details')
	cmd:text()
	cmd:text('Options:')
	
	cmd:option('-mode',			'demo', 'Options: demo | eval')
	cmd:option('-dataset',		'aflw2000', 'Options: aflw2000 | aflwpifa')
	
	cmd:text()
	
	local opt = cmd:parse(arg or {})
	
	return opt 
end

return parse