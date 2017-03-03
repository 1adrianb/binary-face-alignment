require 'torch'
require 'nn'
require 'cudnn'
require 'paths'

require 'bnn'
require 'optim'

require 'gnuplot'
require 'image'
require 'xlua'
local utils = require 'utils'
local opts = require('opts')(arg)

torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local model
if opts.dataset == 'AFLWPIFA' then
	print('Not available for the moment. Support will be added soon')
	os.exit()
	model = torch.load('models/facealignment_binary_pifa.t7')
else
	model = torch.load('models/facealignment_binary_aflw.t7')
end
model:evaluate()

local fileLists = utils.getFileList(opts)
local predictions = {}
local noPoints = 68
if opts.dataset == 'AFLWPIFA' then noPoints = 34; end
local output = torch.CudaTensor(1,noPoints,64,64)

if opts.mode == 'eval' then  
	print('Not available for the moment. Support will be added soon')
	os.exit()
	xlua.progress(0,#fileLists) 
end
for i = 1, #fileLists do	
	local img = image.load(fileLists[i].image)
	local originalSize = img:size()

	img = utils.crop(img, fileLists[i].center, fileLists[i].scale, 256)
	img = img:cuda():view(1,3,256,256)
	
	output:copy(model:forward(img))
	output:add(utils.flip(utils.shuffleLR(opts, model:forward(utils.flip(img)))))

	local preds_hm, preds_img = utils.getPreds(output, fileLists[i].center, fileLists[i].scale)
	
	if opts.mode == 'demo' then
		utils.plot(fileLists[i].image,preds_img:view(noPoints,2),torch.Tensor{originalSize[3],originalSize[2]})
		io.read() -- Wait for user input
	end
	
	if opts.mode == 'eval' then
		predictions[i] = preds_img:clone()
		xlua.progress(i, #fileLists)
	end
end

if opts.mode == 'demo' then gnuplot.closeall() end

if opts.mode == 'eval' then
	predictions = torch.cat(predictions,1)
	local dists = utils.calcDistance(predictions,fileLists)
	utils.calculateMetrics(dists)
end
