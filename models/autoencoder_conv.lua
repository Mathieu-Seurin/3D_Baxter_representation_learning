require 'nn'
local resnet = require("models/resnet")

function getModel(Dimension)
    local Dimension = 3

   nbFilter=32
   -- -- encoder number one-------------- for test
   -- -- NUM_HIDDEN
   -- encoder = nn.Sequential()
   --
   -- encoder:add(nn.SpatialConvolution(3, nbFilter, 3, 3, 1, 1, 1, 1))
   -- encoder:add(nn.SpatialBatchNormalization(nbFilter))
   -- encoder:add(nn.ReLU())
   -- local pool1 = nn.SpatialMaxPooling(2,2,2,2)
   -- encoder:add(pool1)
   --
   -- -- encoder:add(nn.View(-1))
   -- -- encoder:add(nn.Linear(3 * 100 * 100, 100))
   --
   -- -- Create decoder ------------------------------
   -- decoder = nn.Sequential()
   -- -- encoder:add(nn.Linear(100, 3 * 100 * 100))
   -- -- encoder:add(nn.View(3,100,100):setNumInputDims(3))
   -- decoder:add(nn.SpatialMaxUnpooling(pool1))
   -- decoder:add(nn.SpatialConvolution(nbFilter, 3, 3, 3, 1,1,1,1))
   -- -- decoder:add(nn.SpatialBatchNormalization(3))
   -- -- decoder:add(nn.ReLU())
   -- -- decoder number one!!! -----------------------
   encoder = nn.Sequential()

    encoder:add(resnet.getModel(3))
    -- encoder:add(nn.Normalize(2))
    -- print(encoder)

    decoder = nn.Sequential()
    decoder:add(nn.Linear(Dimension,100))
    decoder:add(cudnn.ReLU())
    decoder:add(nn.Linear(100,500))
    decoder:add(cudnn.ReLU())
    local size = 25
    decoder:add(nn.Linear(500,Dimension * size * size))
    decoder:add(nn.View(Dimension,size,size):setNumInputDims(1))

    decoder:add(cudnn.SpatialConvolution(Dimension, nbFilter * 8, 3, 3,1,1,1,1))
    decoder:add(nn.SpatialBatchNormalization(nbFilter * 8))
    decoder:add(cudnn.ReLU())
    decoder:add(nn.SpatialUpSamplingNearest(2))

    decoder:add(cudnn.SpatialConvolution(nbFilter * 8, nbFilter * 4, 3, 3,1,1,1,1))
    decoder:add(nn.SpatialBatchNormalization(nbFilter * 4))
    decoder:add(cudnn.ReLU())
    decoder:add(nn.SpatialUpSamplingNearest(2))

    decoder:add(cudnn.SpatialConvolution(nbFilter * 4, nbFilter * 2, 3, 3,1,1,1,1))
    decoder:add(nn.SpatialBatchNormalization(nbFilter * 2))
    decoder:add(cudnn.ReLU())
    decoder:add(nn.SpatialUpSamplingNearest(2))

    decoder:add(cudnn.SpatialConvolution(nbFilter * 2, nbFilter, 3, 3,1,1,1,1))
    decoder:add(nn.SpatialBatchNormalization(nbFilter))
    decoder:add(cudnn.ReLU())

    decoder:add(cudnn.SpatialConvolution(nbFilter, 3, 3, 3,1,1,1,1))
    -- net:add(nn.SpatialBatchNormalization(3))
   -- -- -- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
   -- -- local method = 'xavier'
   -- -- local encoder = require('weight-init')(encoder, method)

   -- Create autoencoder
   autoencoder = nn.Sequential()
   autoencoder:add(encoder)
   autoencoder:add(decoder)

   return autoencoder

end
