require 'nn'
-- network-------------------------------------------------------
local M = {}
function getModel(Dimension)

   nbFilter=32

   Timnet = nn.Sequential()

   Timnet:add(nn.SpatialConvolution(3, nbFilter, 3, 3))
   Timnet:add(nn.SpatialBatchNormalization(nbFilter))
   Timnet:add(nn.ReLU())
   Timnet:add(nn.SpatialMaxPooling(2,2,2,2))


   Timnet:add(nn.SpatialConvolution(nbFilter, 2*nbFilter, 3, 3))
   Timnet:add(nn.SpatialBatchNormalization(2*nbFilter))
   Timnet:add(nn.ReLU())
   Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

   Timnet:add(nn.SpatialConvolution(2*nbFilter, 4*nbFilter, 3, 3))
   Timnet:add(nn.SpatialBatchNormalization(4*nbFilter))
   Timnet:add(nn.ReLU())
   Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

   Timnet:add(nn.SpatialConvolution(4*nbFilter, 8*nbFilter, 3, 3))
   Timnet:add(nn.SpatialBatchNormalization(8*nbFilter))
   Timnet:add(nn.ReLU())
   Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

   Timnet:add(nn.SpatialConvolution(8*nbFilter, Dimension, 1, 1))
   Timnet:add(nn.SpatialBatchNormalization(Dimension))
   Timnet:add(nn.ReLU())

   -- If image are 200x200 => View(3*10*10):setNumInputDims(3) also works
   Timnet:add(nn.View(-1):setNumInputDims(3))
   Timnet:add(nn.Linear(Dimension*10*10, 500))
   Timnet:add(nn.ReLU())
   Timnet:add(nn.Linear(500, 100))
   Timnet:add(nn.ReLU())
   Timnet:add(nn.Linear(100, Dimension))

   -- Initialisation : "Understanding the difficulty of training deep feedforward neural networks"
   local method = 'xavier'
   local Timnet = require('weight-init')(Timnet, method)
   print('Timnet\n' .. Timnet:__tostring());
   return Timnet
end
M.getModel = getModel
return M
