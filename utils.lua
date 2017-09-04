local utils = {}

-- Transform the coordinates from the original image space to the cropped one
function utils.transform(pt, center, scale, res, invert)
    -- Define the transformation matrix
    local pt_new = torch.ones(3)
    pt_new[1], pt_new[2] = pt[1], pt[2]
    local h = 200*scale
    local t = torch.eye(3)
    t[1][1], t[2][2] = res/h, res/h
    t[1][3], t[2][3] = res*(-center[1]/h+0.5), res*(-center[2]/h+0.5)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_new):sub(1,2):int()
    return new_point
end

-- Crop based on the image center & scale
function utils.crop(img, center, scale, res)
    local l1 = utils.transform({1,1}, center, scale, res, true)
    local l2 = utils.transform({res,res}, center, scale, res, true)

    local pad = math.floor(torch.norm((l1 - l2):float())/2 - (l2[1]-l1[1])/2)
    
    if img:nDimension() < 3 then
      img = torch.repeatTensor(img,3,1,1)
    end

    local newDim = torch.IntTensor({img:size(1), l2[2] - l1[2], l2[1] - l1[1]})
    local newImg = torch.zeros(newDim[1],newDim[2],newDim[3])
    local height, width = img:size(2), img:size(3)

    local newX = torch.Tensor({math.max(1, -l1[1]+1), math.min(l2[1], width) - l1[1]})
    local newY = torch.Tensor({math.max(1, -l1[2]+1), math.min(l2[2], height) - l1[2]})
    local oldX = torch.Tensor({math.max(1, l1[1]+1), math.min(l2[1], width)})
    local oldY = torch.Tensor({math.max(1, l1[2]+1), math.min(l2[2], height)})

    newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))

    newImg = image.scale(newImg,res,res)
    return newImg
end

function utils.getPreds(heatmaps, center, scale)
    if heatmaps:nDimension() == 3 then heatmaps = heatmaps:view(1, unpack(heatmaps:size():totable())) end

    -- Get locations of maximum activations
    local max, idx = torch.max(heatmaps:view(heatmaps:size(1), heatmaps:size(2), heatmaps:size(3) * heatmaps:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % heatmaps:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(heatmaps:size(3)):floor():add(1)

    for i = 1,preds:size(1) do        
        for j = 1,preds:size(2) do
            local hm = heatmaps[{i,j,{}}]
            local pX, pY = preds[{i,j,1}], preds[{i,j,2}]
            if pX > 1 and pX < 64 and pY > 1 and pY < 64 then
                local diff = torch.FloatTensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
                preds[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    preds:add(-0.5)

    -- Get the coordinates in the original space
    local preds_orig = torch.zeros(preds:size())
    for i = 1, heatmaps:size(1) do
        for j = 1, heatmaps:size(2) do
            preds_orig[i][j] = utils.transform(preds[i][j],center,scale,heatmaps:size(3),true)
        end
    end
    return preds, preds_orig
end

function utils.shuffleLR(opts, x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

	local matched_parts = nil
	if opts.dataset == 'AFLWPIFA' then
		matched_parts = {
			{1,17},   {2,16},   {3,15},
            {4,14}, {5,13}, {6,12}, {7,11}, {8,10},
            {18,27},{19,26},{20,25},{21,24},{22,23},
            {37,46},{38,45},{39,44},{40,43},
            {42,47},{41,48},
            {32,36},{33,35},
			{51,53},{50,54},{49,55},{62,64},{61,65},{68,66},{60,56},
            {59,57}
		}		
	else
		matched_parts = {
			{1,17},   {2,16},   {3,15},
            {4,14}, {5,13}, {6,12}, {7,11}, {8,10},
            {18,27},{19,26},{20,25},{21,24},{22,23},
            {37,46},{38,45},{39,44},{40,43},
            {42,47},{41,48},
            {32,36},{33,35},
			{51,53},{50,54},{49,55},{62,64},{61,65},{68,66},{60,56},
            {59,57}
		}
	end

    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

function utils.flip(x)
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end

function utils.calcDistance(predictions,groundTruth)
  local n = predictions:size()[1]
  gnds = torch.Tensor(n,16,2)
  for i=1,n do
    gnds[{{i},{},{}}] = groundTruth[i].points
  end

  local dists = torch.Tensor(predictions:size(2),predictions:size(1))
  -- Calculate L2
	for i = 1,predictions:size(1) do
		for j = 1,predictions:size(2) do
			if gnds[i][j][1] > 1 and gnds[i][j][2] > 1 then
				dists[j][i] = torch.dist(gnds[i][j],predictions[i][j])/groundTruth[i].headSize
			else
				dists[j][i] = -1
			end
		end
	end

  return dists
end

--http://stackoverflow.com/questions/640642/how-do-you-copy-a-lua-table-by-value
function table.copy(t)
   if t == nil then
      return {}
   end
   local u = { }
   for k, v in pairs(t) do u[k] = v end
   return setmetatable(u, getmetatable(t))
end

-- originally created in torch dp package, by nicholas leonard
function torch.swapaxes(tensor, new_axes)

   -- new_axes : A table that give new axes of tensor, 
   -- example: to swap axes 2 and 3 in 3D tensor of original axes = {1,2,3}, 
   -- then new_axes={1,3,2}
 
   local sorted_axes = table.copy(new_axes)
   table.sort(sorted_axes)
   
   for k, v in ipairs(sorted_axes) do
      assert(k == v, 'Error: new_axes does not contain all the new axis values')
   end       

   -- tracker is used to track if a dim in new_axes has been swapped
   local tracker = torch.zeros(#new_axes)   
   local new_tensor = tensor

   -- set off a chain swapping of a group of intraconnected dimensions
   _chain_swap = function(idx)
      -- if the new_axes[idx] has not been swapped yet
      if tracker[new_axes[idx]] ~= 1 then
         tracker[idx] = 1
         new_tensor = new_tensor:transpose(idx, new_axes[idx])
         return _chain_swap(new_axes[idx])
      else
         return new_tensor
      end    
   end
   
   for idx = 1, #new_axes do
      if idx ~= new_axes[idx] and tracker[idx] ~= 1 then
         new_tensor = _chain_swap(idx)
      end
   end
   
   return new_tensor
end

function utils.bounding_box(iterable)
    local mins = torch.min(iterable, 1):view(2)
    local maxs = torch.max(iterable, 1):view(2)

	local center = torch.FloatTensor{maxs[1]-(maxs[1]-mins[1])/2, maxs[2]-(maxs[2]-mins[2])/2}
    
	return center, (maxs[1]-mins[1]+maxs[2]-mins[2])/190 --center and scale
end

local function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

function utils.getFileList(opts)
	opts.dataset = string.upper(opts.dataset)
	local fileLists = {}
	for f in paths.files('dataset/'..opts.dataset,".jpg") do
		-- Construct the content
		local orig_pts = torch.load('dataset/'..opts.dataset..'/'..f:sub(1,#f-4)..'.t7')
		
		if orig_pts ~= nil then 
			local pts = torch.swapaxes(orig_pts.pt3d_68[{{1,2},{}}],{2,1})
			local center, scale = utils.bounding_box(pts)
			
			local dataPts = {}
			dataPts.center = center
			dataPts.scale = scale
			dataPts.image = 'dataset/'..opts.dataset..'/'..f
			
			fileLists[#fileLists+1] =  dataPts
		end
		if (opts.mode == 'demo' and #fileLists==100) then fileLists=subrange(fileLists,90,100); break end
	end

	return fileLists
end

-- Requires gnuplot
function utils.plot(surface, points, size)
	if points:nDimension()~=2 then
		points = points:view(points:size(2),2)
	end
	
    gnuplot.figure(1)
    gnuplot.raw("set size ratio -1")
	gnuplot.raw("set xrange [0:"..size[1].."]")
	gnuplot.raw("set yrange [0:"..size[2].."]")
    gnuplot.raw("unset key; unset tics; unset border;")
	gnuplot.raw("set multiplot layout 1,1 margins 0.05,0.95,.1,.99 spacing 0,0")
    gnuplot.raw("plot '"..surface.."' binary filetype=jpg with rgbimage")  

	gnuplot.raw(" set yrange ["..size[2]..":0] ") 
	
	local x = points[{{},{1}}]:contiguous():view(68)
	local y = points[{{},{2}}]:contiguous():view(68)

	gnuplot.plot(x, y, '+')
	gnuplot.raw("unset multiplot")
end

local function displayPCKh(dists, idxs, title, disp_key)
	local xs = torch.linspace(0,0.5,30)
	local ys = torch.zeros(xs:size(1))
	local total = {dists[{idxs[1],{}}]:gt(-1):sum(),
					dists[{idxs[2],{}}]:gt(-1):sum()}
	for i = 1, xs:size(1) do
		ys[i] = 0.5*((dists[{idxs[1],{}}]:lt(xs[i]):sum()-(dists:size(2)-total[1]))/total[1]+(dists[{idxs[2],{}}]:lt(xs[i]):sum()-(dists:size(2)-total[2]))/total[2])
	end

	local command = {xs,ys,'-'}
	gnuplot.raw('set title "'..title..'"')
	if not disp_key then 
		gnuplot.raw('unset key')
	else
		gnuplot.raw('set key font ",6" right bottom')
	end
	gnuplot.raw('set xrange [0:0.5]')
	gnuplot.raw('set yrange [0:1]')
	gnuplot.plot(unpack(command))
end

function utils.calculateMetrics(dists)
	gnuplot.raw('set bmargin 1')
	gnuplot.raw('set lmargin 3.2')
	gnuplot.raw('set rmargin 2')
	gnuplot.raw('set multiplot layout 2,3 title "MPII Validation (PCKh)"')
	gnuplot.raw('set xtics font ",6"')
	gnuplot.raw('set xtics font ",6"')
	displayPCKh(dists, {9,10}, 'Head')
	displayPCKh(dists, {2,5}, 'Knee')
	displayPCKh(dists, {1,6}, 'Ankle')
	gnuplot.raw('set tmargin 2.5')
	gnuplot.raw('set bmargin 1.5')
	displayPCKh(dists, {13,14}, 'Shoulder')
	displayPCKh(dists, {12,15}, 'Elbow')
	displayPCKh(dists, {11,16}, 'Wrist', true)	
	gnuplot.raw('unset multiplot')
	
    local threshold = 0.5
    dists:apply(function(x)
        if x>=0 and x<= threshold then 
            return 1
        elseif x>threshold then 
            return 0
        end
    end)

    local count = torch.zeros(16)
    local sums = torch.zeros(16)
    for i=1,16 do
        dists[i]:apply(function(x)
            if x ~= -1 then
                count[i] = count[i] + 1
                sums[i] = sums[i] + x
            end
        end)
    end

    local partNames = {'Head', 'Knee', 'Ankle', 'Shoulder', 'Elbow', 'Wrist', 'Hip'}
    local partsC =  torch.Tensor({{9,10},{2,5},{1,6},{13,14},{12,15},{11,16},{3,4}})
    print('PCKh results:')
    for i=1,#partNames do
        print(partNames[i]..': ',(sums[partsC[i][1]]/count[partsC[i][1]]+sums[partsC[i][2]]/count[partsC[i][1]])*100/2)
    end
end

return utils
