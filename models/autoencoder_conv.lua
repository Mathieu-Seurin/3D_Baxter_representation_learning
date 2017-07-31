require 'nn'
local resnet = require("models/resnet")

function getModel(Dimension)
   local Dimension = 3

   nbFilter=32
   encoder = nn.Sequential()

    -- encoder:add(resnet.getModel(Dimension))
    encoder = resnet.getModel(Dimension)
    -- print(encoder)

    -- net:add(nn.SpatialBatchNormalization(3))
   -- -- -- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
   -- -- local method = 'xavier'
   -- -- local encoder = require('weight-init')(encoder, method)

   decoder = timnet_r(Dimension)

   -- Create autoencoder
   autoencoder = nn.Sequential()
   autoencoder:add(encoder)
   autoencoder:add(decoder)


   return autoencoder

end




-- original
function custom(Dimension)
    decoder = nn.Sequential()
    decoder:add(nn.Linear(Dimension,100))
    decoder:add(cudnn.ReLU())
    decoder:add(nn.Linear(100,500))
    decoder:add(cudnn.ReLU())
    local size = 28
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

    decoder:add(cudnn.SpatialConvolution(nbFilter * 2, nbFilter * 2, 3, 3,1,1,1,1))
    decoder:add(nn.SpatialBatchNormalization(nbFilter * 2))
    decoder:add(cudnn.ReLU())
    decoder:add(nn.SpatialUpSamplingNearest(2))

    decoder:add(cudnn.SpatialConvolution(nbFilter * 2, nbFilter, 3, 3,1,1,1,1))
    decoder:add(nn.SpatialBatchNormalization(nbFilter))
    decoder:add(cudnn.ReLU())

    decoder:add(cudnn.SpatialConvolution(nbFilter, nbFilter, 3, 3,1,1,1,1))
    decoder:add(nn.SpatialBatchNormalization(nbFilter))
    decoder:add(cudnn.ReLU())

    decoder:add(cudnn.SpatialConvolution(nbFilter, 3, 3, 3,1,1,1,1))
    return decoder
end

function timnet_r(Dimension)
    decoder = nn.Sequential()
    decoder:add(nn.Linear(Dimension,100))
    decoder:add(cudnn.ReLU())
    decoder:add(nn.Linear(100,500))
    decoder:add(cudnn.ReLU())
    local size = 28
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
    -- local method = 'xavier'
    -- local encoder = require('weight-init')(decoder, method)
    return decoder
end
