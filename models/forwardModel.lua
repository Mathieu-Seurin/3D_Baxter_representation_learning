-- tnt = require 'torchnet'
-- vision = require 'torchnet-vision'
-- nngraph = require 'nngraph'
-- in order to provide a prediction of next state, we need to predict the reward of
--current state (given the state and the action taken).
--This is a state, action, reward and successor (s, a, r, s’) transition standard to RL
--that admits many kinds of self-supervision \cite{Shelhamer17}

--This concrete model predicts next state and next reward given the input image,
--its state and action performed
--loss function used is the difference between the predicted next state and real next state.
--( using nn.MSECriterion)
---------------------------------------------------------------

-- function getModel(Dimension)
--
--    whole_net = nn.Sequential()
--
--    model = vision.models.inceptionv4
--    pretrain_net = model.load()
--    pretrain_net:evaluate()
--
--
--    pretrain_net.modules[21] = nil --nn.SoftMax
--    pretrain_net.modules[20] = nil --nn.Linear(1536 -> 1001)
--    --pretrain_net.modules[19] is a View(1536)
--
--    for i=1,19 do
--       c = pretrain_net:get(i)
--       c.updateGradInput = function(self, inp, out) end
--       c.accGradParameters = function(self,inp, out) end
--    end
--
--    whole_net:add(pretrain_net)
--
--    --whole_net:add(nngraph.)
--    whole_net:add(nn.Linear(1536,Dimension))
--
--    return whole_net
-- end


local M = {}

function file_exists(name)
   --tests whether the file can be opened for reading
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

local function patch(nn_module)
   if nn_module.modules then
      for i =1,#nn_module.modules do
         patch(nn_module.modules[i])
      end
   else
      nn_module.accGradParameters = function(self,inp, out) end  -- this is freezing FROZEN_LAYER
   end
end

function getModel(Dimension)

   local whole_net, pretrain_net

   whole_net = nn.Sequential()

   local model = "resnet-"..RESNET_VERSION..".t7"
   local model_full_path = "./models/"..model

   if file_exists(model_full_path) then
      pretrain_net = torch.load(model_full_path)
   else
      print(model_full_path)
      error([[------------------The above model was required but it doesn't exist,
      download it here :\n https://github.com/facebook/fb.resnet.torch/tree/master/pretrained \nAnd put it in models/ as resnet-VERSION.t7
      Ex : resnet-18.t7 -------------------]])
   end

   if RESNET_VERSION == 18 or RESNET_VERSION == 34 then
      pretrain_net.modules[11] = nil --nn.Linear(512 -> 1000)
      --pretrain_net.modules[10] is a View(512)

   else
      error("Version of resnet not known or not available")
   end

   -- Block backpropagation, i.e Freeze FROZEN_LAYER layers (defined in hyperparams.lua)
   for i=1,FROZEN_LAYER do
      nn_module = pretrain_net:get(i)
      patch(nn_module)  -- Freezes
   end

   whole_net:add(pretrain_net)
   whole_net:add(nn.Linear(512,Dimension))

   whole_net:evaluate()

   return whole_net
end

M.getModel = getModel

return M
