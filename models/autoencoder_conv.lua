require 'nn'

function getModel()

   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
   encoder:add(nn.ReLU(true))

   local pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1)
   encoder:add(pool1)

   encoder:add(nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1))
   encoder:add(nn.ReLU(true))

   local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1)
   encoder:add(pool2)
   
   encoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
   encoder:add(nn.ReLU(true))

   local pool3 = nn.SpatialMaxPooling(2, 2, 2, 2)
   encoder:add(pool3)

   encoder:add(nn.View(8*25*25):setNumInputDims(3))
   encoder:add(nn.Linear(8*25*25,NUM_HIDDEN))

   -- Create decoder
   decoder = nn.Sequential()
   decoder:add(nn.Linear(NUM_HIDDEN,8*25*25))
   decoder:add(nn.View(8,25,25):setNumInputDims(1))
   decoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.ReLU(true))

   decoder:add(nn.SpatialMaxUnpooling(pool3))
   --decoder:add(nn.SpatialUpSamplingNearest(2))

   decoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.ReLU(true))

   decoder:add(nn.SpatialMaxUnpooling(pool2))
   --decoder:add(nn.SpatialUpSamplingNearest(2))

   decoder:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.ReLU(true))

   decoder:add(nn.SpatialMaxUnpooling(pool1))
   --decoder:add(nn.SpatialUpSamplingNearest(2))

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


