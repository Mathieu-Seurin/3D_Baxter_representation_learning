tnt = require 'torchnet'
vision = require 'torchnet-vision'

function getModel(Dimension)

   model = vision.models.inceptionv4
   pretrain_net = model.load()
   pretrain_net:evaluate()

   pretrain_net.modules[21] = nil --nn.SoftMax
   pretrain_net.modules[20] = nil --nn.Linear(1536 -> 1001)
   --pretrain_net.modules[19] is a View(1536)

   frozen_part = 15
   
   for i=1,frozen_part do
      c = pretrain_net:get(i)
      c.updateGradInput = function(self, inp, out) end
      c.accGradParameters = function(self,inp, out) end
   end

   pretrain_net:add(nn.Linear(1536,Dimension))
   return pretrain_net
end
           

   
   
