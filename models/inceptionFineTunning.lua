tnt = require 'torchnet'
vision = require 'torchnet-vision'

function getModel(Dimension)

   whole_net = nn.Sequential()
   
   model = vision.models.inceptionv4
   pretrain_net = model.load()
   pretrain_net:evaluate()

   
   pretrain_net.modules[21] = nil --nn.SoftMax
   pretrain_net.modules[20] = nil --nn.Linear(1536 -> 1001)
   --pretrain_net.modules[19] is a View(1536)

   for i=1,19 do
      c = pretrain_net:get(i)
      c.updateGradInput = function(self, inp, out) end
      c.accGradParameters = function(self,inp, out) end
   end

   whole_net:add(pretrain_net)
   whole_net:add(nn.Linear(1536,Dimension))
   
   return whole_net
end
           

   
   
