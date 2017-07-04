require 'nn'

function getModel()

   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
   encoder:add(nn.ReLU(true))

   encoder:add(nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1))
   encoder:add(nn.ReLU(true))
   
   encoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
   encoder:add(nn.ReLU(true))

   encoder:add(nn.View(8*25*25):setNumInputDims(3))
   encoder:add(nn.Linear(8*25*25,NUM_HIDDEN))

   -- Create decoder
   decoder = nn.Sequential()
   decoder:add(nn.Linear(NUM_HIDDEN,8*25*25))
   decoder:add(nn.View(8,25,25):setNumInputDims(1))
   decoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.ReLU(true))

   decoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.ReLU(true))

   decoder:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.ReLU(true))

   decoder:add(nn.SpatialConvolution(16, 3, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.View(3, 200, 200))

   -- -- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
   -- local method = 'xavier'
   -- local encoder = require('weight-init')(encoder, method)
   
   -- Create autoencoder
   autoencoder = nn.Sequential()
   autoencoder:add(encoder)
   autoencoder:add(decoder)

   return autoencoder
   
end
