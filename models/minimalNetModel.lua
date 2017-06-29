require 'nn'

-- network-----mini size net for cpu test runs only--------------------------------------------------
-- function getModel()
-- 	nbFilter=1
-- 	input = nn.Identity()()
-- 	conv1 = nn.SpatialConvolution(3, nbFilter, 3, 3)(input)
-- 	--1*198*198, (one feature Map of 200*200, reduced to 198*198 by padding of convolution):
-- 	View_1=nn.View(1*198*198)(conv1)
-- 	out=nn.ReLU()(nn.Linear(1*198*198, 3)(View_1))
-- 	gmod = nn.gModule({input}, {out})
-- 	local method = 'xavier'
-- 	local gmod = require('weight-init')(gmod, method)
-- 	return gmod
-- end

function getModel(Dimension)
	nbFilter = 12
	n2 = 3
	alpha = 0.1
	net = nn.Sequential()
	-- https://github.com/torch/nn/blob/master/doc/convolution.md#spatialconvolution
	net:add(nn.SpatialConvolution(3, nbFilter, 3, 3, 1,1, (3-1)/2, (3-1)/2))
	net:add(nn.ReLU())
	-- net:add(nn.ELU(alpha))
	net:add(nn.SpatialMaxPooling(2,2,2,2))

	net:add(nn.SpatialConvolution(nbFilter, n2, 3, 3, 1,1, (3-1)/2, (3-1)/2))
	net:add(nn.ReLU())
	-- net:add(nn.ELU(alpha))
	net:add(nn.SpatialMaxPooling(2,2,2,2))

	net:add(nn.View(n2 * 50 * 50))
	net:add(nn.Linear(n2 * 50 * 50, 100))
	net:add(nn.ReLU())
	-- net:add(nn.ELU(alpha))
	net:add(nn.Linear(100, Dimension))
	local method = 'xavier'
	local net = require('weight-init')(net, method)
	-- print("Simple Net: \n" .. net:__tostring());
	return net
end
