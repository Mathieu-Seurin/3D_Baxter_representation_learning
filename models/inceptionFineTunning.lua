tnt = require 'torchnet'
vision = require 'torchnet-vision'

function getModel(Dimension)

   wholeNet = nn.Sequential()

   model = vision.models.inceptionv4
   pretrain_net = model.load()
   pretrain_net:evaluate()

   pretrain_net.modules[21] = nil --nn.SoftMax
   pretrain_net.modules[20] = nil --nn.Linear(1536 -> 1001)

   wholeNet:add(pretrain_net)

   --pretrain_net.modules[19] is a View(1536)

   wholeNet:add(nn.Linear(1536,Dimension))
   return wholeNet
end
           

   
   
