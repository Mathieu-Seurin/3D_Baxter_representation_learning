require 'nngraph'

DIMENSION_IN = 2 --Default for mobileRobot
DIMENSION_OUT = 3 --Default for mobileRobot
LR = 0.05
NB_EPOCHS = 10 --1000
WEIGHT_IMPORTANCE_FOR_FIRST_INPUT = 0.5
DATA_FOLDER = 'mobileRobot'
-- REWARD_INDEX = 1  --3 reward values: -1, 0, 10

--local pred = model:forward(inputs)
-- local firstLoss = criterion:forward(pred[1], first)
-- local secondLoss = criterion:forward(pred[2], second)
-- local firstGrad = criterion:backward(pred[1], first)
-- local secondGrad = criterion:backward(pred[2], second)
-- local grad = {firstGrad,secondGrad}
-- model:backward(inputs, grad)

--The nn methods such as train rely on nn torch tutorial in
--https://github.com/AILabUSiena/NeuralNetworksForBeginners/tree/master/torch/xor
--https://github.com/torch/nngraph#a-network-with-2-inputs-and-2-outputs
--Todo: weight decay https://arxiv.org/pdf/1703.05298.pdf

function printNetworkGraph(gmodule)
    -- gmod is what we send to forward and backward pass
    graph.dot(gmodule.fg, 'Forward Graph')
    graph.dot(gmodule.bg, 'Backward Graph')
    -- param, grad = net:getParameters() --returns all weights and the gradients of the network in two 2D vector
    -- print('params: ',param, 'grad: ',grad)
    -- plot_loss()
end

function sequentialTorchNNExample()
   mlp= nn.Sequential();       --Create a network that takes a Tensor as input
   mlp:add(nn.SplitTable(2))
   c=nn.ParallelTable()      --The two Tensors go through two different Linear
   c:add(nn.Linear(10,3))	   --Layers in Parallel
   c:add(nn.Linear(10,7))
   mlp:add(c)                 --Outputing a table with 2 elements
   p=nn.ParallelTable()      --These tables go through two more linear layers
   p:add(nn.Linear(3,2))	   -- separately.
   p:add(nn.Linear(7,1))
   mlp:add(p)
   mlp:add(nn.JoinTable(1))   --Finally, the tables are joined together and output.

   pred=mlp:forward(torch.randn(10,2))
   print("prediction: ")
   print(pred)

   for i=1,NB_EPOCHS do             -- A few steps of training such a network..
      x= torch.ones(10,2);
      y= torch.Tensor(3); y:copy(x:select(2,1,1):narrow(1,1,3))
      pred= mlp:forward(x)

      criterion= nn.MSECriterion()
      --FORWARD takes two params:  if we pass torch.randn(2,5), That input goes
      -- into each of the nn.Linear(3,5) and nn.Linear(5,5) defined inputs, If you pass torch.randn(2, 5),
      -- it is a vector of size 5 (with mini-batch size 2)
      local err = criterion:forward(pred, y) --
      local gradients = criterion:backward(pred,y);
    --   print('x and y: ')
    --   print(x,y)
      mlp:zeroGradParameters();
      mlp:backward(x, gradients);
      mlp:updateParameters(LR);  --exclusive of nn?
      print('Error (MSE):')
      print(err)
   end
end

------------
--Torch.nngraph Version: more modularity and power than Torch.nn
-- TEST
------------
function twoInputs2OutputsNNGraph()
    print('2 input 2 output model')
    h1 = nn.Linear(20, 20)() --Linear takes in a vector of size 20
    h2 = nn.Linear(10, 10)()
    hh1 = nn.Linear(20, 1)(nn.Tanh()(h1)) -- equvalent to applying left to right or what in Python would be Linear(Tanh(h1))
    hh2 = nn.Linear(10, 1)(nn.Tanh()(h2))
    madd = nn.CAddTable()({hh1, hh2})
    oA = nn.Sigmoid()(madd)
    oB = nn.Tanh()(madd)
    gmod = nn.gModule({h1, h2}, {oA, oB})

    x1 = torch.rand(20)
    x2 = torch.rand(10)

    gmod:updateOutput({x1, x2})
    gmod:updateGradInput({x1, x2}, {torch.rand(1), torch.rand(1)})
    --graph.dot(gmod.fg, 'Big MLP 2I2O')
    -- Alternatively, you can use - to make your code looks like the data flow:
    -- h1 = - nn.Linear(20,20)
    -- h2 = - nn.Linear(10,10)
    -- hh1 = h1 - nn.Tanh() - nn.Linear(20,1)
    -- hh2 = h2 - nn.Tanh() - nn.Linear(10,1)
    -- madd = {hh1,hh2} - nn.CAddTable()
    -- oA = madd - nn.Sigmoid()
    -- oB = madd - nn.Tanh()
    -- gmod = nn.gModule( {h1,h2}, {oA,oB} )
end

---------------------------------------------------------------------------------------
-- Function :Uses Torch.nn Version, from https://arxiv.org/pdf/1703.05298.pdf
-- forward(input) returns the output of the multi layer perceptron w.r.t the given input; it updates
-- the input/output states variables of each modules, preparing the network for the backward
-- step; its output will be immediately passed to the loss function to compute the error.
-- zeroGradParameters() resets to null values the state of the gradients of the all the parameters.
-- backward(gradients) actually computes and accumulates (averaging them on the number of samples)
-- the gradients with respect to the weights of the network, given the data in input and
-- the gradient of the loss function.
-- updateParameters(learningrate) modifies the weights according to the Gradient Descent procedure
-- using the learning rate as input argument.
-- Output ()
----
-- @misc{neural_networks_for_beginners,
-- Author = {Francesco Giannini and Vincenzo Laveglia and Alessandro Rossi and Dario Zanca and Andrea Zugarini},
-- Title = {Neural Networks for Beginners. A fast implementation in Matlab, Torch, TensorFlow},
-- Year = {2017},
-- Eprint = {arXiv:1703.05298},
-- }
    --TODO    --
    --A simple check on the minimum value of the absolute values of gradients
    -- saved in grad can be used to stop the training procedure.
    -- Another regularization method can be accomplished by implementing the weight decay method
---------------------------------------------------------------------------------------
-- function trainNN(net, x, y, epochs, LR)
--     print('Torch.NN Train: x and y and sizes: ',x, y, x:size(), y:size())
--     criterion = nn.MSECriterion()
--     local loss = torch.Tensor(epochs):fill(0) -- training loss initialization
--     for i=1, epochs do             -- A few steps of training such a network..
--        pred = net:forward(x) --network forward step
--        loss[i] = criterion:forward(pred, y) -- network error evaluation (output, target)
--        local gradients = criterion:backward(pred, y); --loss gradients
--        net:zeroGradParameters(); -- zero reset of gradients
--        net:backward(x, gradients); --network backward step  (outputs, target)
--        net:updateParameters(LR); --given the LR, update parameters
--        print('Error (MSE):', loss[i])
--     end
--     return net:evaluate()  --Needed in nngraph too? only for sequential models?
-- end

function printAll(t, name)
    print(name)
    print (type(t), #t)
end
----------------
------TORCH.NNGraph VERSION
----------------
-- Uses a typical generic gradient update function
function updateGradient(net, x, y, criterion, learningRate)
    printAll(x,'x')
    local y_pred = net:forward(x) --network forward step
    print("prediction: ", y_pred)
    local loss = criterion:forward(y_pred, y) -- -- network error evaluation (output, target)
    local gradients = criterion:backward(y_pred, y) --loss gradients
    net:zeroGradParameters() -- zero reset of gradients,  --sets them to zero, needs to be done in each fwd and backward update, otherwise Torch internally accumulates the sum of all gradients
    net:backward(x, gradients) --network backward step  (outputs, target)
    net:updateParameters(LR) --given the LR, update parameters
    return loss
end


--Uses 2 losses, one per output being optimized
function trainSeveralLossesNNGraph(net, input, output, epochs, LR, weightImportanceForFirstInput)
    --https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/8
    -- multiCriterion = nn.MultiCriterion() -- returns a Criterion which is a weighted sum of other Criterion.
    -- Criterions are added using the method :add,
    -- where weight is a scalar (default 1). Each criterion is applied to the same input and target.
    local MSECrit = nn.MSECriterion() --.cuda()
    --https://github.com/torch/nn/blob/master/doc/criterion.md#nn.CrossEntropyCriterion
    --you don't put any activation unit at the end of you network
    -- and this criterion calculate the logsoftmax and the classification loss
    local crossEntrCrit = nn.CrossEntropyCriterion(weights)
    crossEntrCrit.nll.sizeAverage = false
    local multiCrit = nn.MultiCriterion():add(MSECrit, weightImportanceForFirstInput):add(crossEntrCrit) -- avged sum of criteria
    --alternatively:    --loss = MSECrit(x1, label1) + MSECrit(x2, label2)
    --loss:backward()  --vs net:backward(x, gradients)?

    local loss = torch.Tensor(epochs):fill(0) -- training loss initialization
    for i=1, epochs do             -- A few steps of training such a network..
       loss[i] = updateGradient(net, input, output, multiCrit, LR)
       print('Error (MSE loss):', loss[i])
    end
    net:evaluate() --Should be done only once per whole training, needed really only if doing dropout
end

--1 unique loss
function trainNNGraph(net, input, output, epochs, LR)
    local MSECrit = nn.MSECriterion() --.cuda()
    print('trainNNGraph: input, output and sizes: ',input, output, input:size(), output:size())
    local loss = torch.Tensor(epochs):fill(0) -- training loss initialization
    for i=1, epochs do             -- A few steps of training such a network..
       loss[i] = updateGradient(net, input, output, MSECrit, LR)
       print('Error (MSE loss):', loss[i])
    end
    net:evaluate() --Should be done only once per whole training, needed really only if doing dropout
end
---------------------
--Torch.nngraph Version Network
-- state_out_dim is a finetunable param. Initially, as in Jonchowscki, 2
--
---------------------
function getStateAndAction2stateAndRewardNetwork(state_in_dim, state_out_dim)
    -- Input: State representation (s_t) and action  TODO: nb_neurons per layer, e.g. 32 in TimNet, use dropout as in https://github.com/Element-Research/rnn/issues/243  ?
    -- Output: State representation (next state s_t+1) and reward
    --state_out_dim = nb_filter = hyperparam, we keep it simple to 2 right now
    -- size mismatch, m1: [8 x 2], m2: [1 x 2] at /tmp/luarocks_torch-scm-1-1606/torch7/lib/TH/generic/THTensorMath.c:1293
    -- ReLU in pyTorch:  import torch.nn.functional as F .. x = F.relu( Conv2d(20, 20, 5) | ReLU in Torch: nn.ReLU
    -- nn.HardShrink applies the hard shrinkage function element-wise to the input Tensor. lambda is set to 0.5 by default.
    -- HardShrinkage operator is defined as:
    --        ⎧ x, if x >  lambda
    -- f(x) = ⎨ x, if x < -lambda
    --        ⎩ 0, otherwise
    --May need extra layer for modularity:  https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion
    --See https://github.com/torch/nn/blob/master/doc/table.md#nn.ConcatTable
    -- ConcatTable applies each member module to the same input Tensor and outputs a table
    -- A)
    inStateT = nn.Identity()()
    inAction = nn.Identity()()
    --hidd_state = nn.Linear(state_in_dim, state_out_dim)(inStateT)
    stateAndAction = nn.ConcatTable()()-- nn.Concat(1) -- dim throuh which inputs are concatenated: inAction, stateT)
    stateAndAction:add(inStateT)
    stateAndAction:add(inAction) -- Action does not need to be encoded nor perturbed, it needs to help predict the reward
    --hidd_action = nn.Linear(state_in_dim, state_out_dim)(inAction)
    -- B)
    -- inStateT = nn.Linear(state_in_dim, state_out_dim)() -- Linear layer with input size 1 and output size 2
    -- inAction = nn.Linear(state_in_dim, state_out_dim)()
    -- outStateT1 = nn.ReLU()(nn.Linear(state_out_dim, state_out_dim)(inStateT)) -- equivalent to applying left to right or what in Python would be Linear(Tanh(h1))
    -- outRewardT = nn.HardShrink(0)(nn.Linear(state_out_dim, state_in_dim)(inAction)) --NOTE Reward T or T+1?
    -- --stateAndAction = nn.Concat()({hh1, hh2}) --Combining them by  CONCAT? before or after the activ function?
    print(stateAndAction)--, 'stateAndAction')
    outStateT1 = nn.Linear(state_out_dim, state_out_dim)(stateAndAction)-- equivalent to applying left to right or what in Python would be Linear(Tanh(h1))
    --outRewardT = nn.Linear(state_out_dim, 1)(hidd_state) --NOTE Reward T or T+1? (in any case, 1 output numeric value)
    --option B: --Predict reward based on direct state or encoded hidd_state?
    outRewardT = nn.Linear(state_in_dim, 1)(inStateT)
    --it in a 2 tensor input 2 tensor output network to create a bottle neck -- nn.Concat concatenates the output of one layer of "parallel" modules along the provided dimension"
    --outState = nn.ReLU()(stateAndAction) --nn.Sigmoid()(stateAndAction)
    --No need for any activ. function at all because our states should not be constrained in any range outReward = nn.HardShrink(0)(stateAndAction)
    --For rewards of -1, 0 and 10 ---TODO: add xavier weight initialization? https://github.com/Mathieu-Seurin/baxter_representation_learning_3D/blob/master/models/topUniqueSimplerWOTanh.lua
    gmod = nn.gModule({inStateT, inAction}, {outStateT1, outRewardT}) -- Parameters are Input and Output to our network  --Alterhative to nn.JoinTable?
    --printNetworkGraph(gmod)
    return gmod
end


-- function getImageAndAction2stateAndRewardNetwork(DIMENSION_OUT)
--     -- Input: State representation (s_t) and action
--     -- Output: State representation (next state s_t+1) and reward
--     --TOOD qlua: /home/natalia/torch/install/share/lua/5.2/nn/Linear.lua:66: size mismatch, m1: [8 x 2], m2: [20 x 20]
--     -- parents for nodes that do computation. Here is the same addition example:
--     -- x1 = nn.Identity()() --NOTE: When to use input Identity vs input Layer directly such as Linear?
--     -- x2 = nn.Identity()()
--
--     inImg = nn.Linear(1, 2)()
--     inAction = nn.Linear(1, 2)()
--     hh1 = nn.Linear(2, 1)(nn.Tanh()(inImg)) -- equivalent to applying left to right or what in Python would be Linear(Tanh(h1))
--     hh2 = nn.Linear(2, 1)(nn.Tanh()(inAction)) --NOTE Linear layers always map to 1 as output in second param?
--     -- Tanh activation function because our rewards can be in [-1, 1], if in (0, 1) use Sigmoid?
--     statePlusRw = nn.Concat()({hh1, hh2}) --Combining them by  CONCAT?
--
--     --	 nn.Concat, passes the same input to all the parallel branches.
--     outState = nn.Sigmoid()(statePlusRw)
--     outReward = nn.Tanh()(statePlusRw)
--     gmod = nn.gModule({inImg, inAction}, {outState, outReward}) -- Parameters are Input and Output to our network  --Alterhative to nn.JoinTable?
--     -- gmod is what we send to forward and backward pass
--     return gmod
-- end

-------------------------------------------------------------------------------


--sequentialTorchNNExample()
--train(n, x, y, NB_EPOCHS) --requires generating XOR dataset
twoInputs2OutputsNNGraph()


x1 = torch.rand(20)
x2 = torch.rand(10)


--First NeuralNetworksForBeginners
local statesT = torch.Tensor{{0,1},{1,1}, {1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}}
--clockwise from 12 o'clock to 10.30, where the reward finally is
local actions = torch.Tensor{{0,1},{1,1}, {1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}}
local statesT1 = torch.Tensor{{1,1}, {1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1}}
local rewards = torch.Tensor{0,0,0,0,0,0,0,1} --torch.rand(10)
n = getStateAndAction2stateAndRewardNetwork(DIMENSION_IN, DIMENSION_OUT)
trainSeveralLossesNNGraph(n, {statesT, actions}, {statesT1, rewards}, NB_EPOCHS, LR, WEIGHT_IMPORTANCE_FOR_FIRST_INPUT)

x=torch.ones(10,2);  -- or torch.randn(10,2)
y=torch.Tensor(3); y:copy(x:select(2,1,1):narrow(1,1,3)) --narrow

-- 2nd Networks--
local imgs = torch.Tensor{{1,2,3},{4,5,6},{7,8,9}}
-- m = getImageAndAction2stateAndRewardNetwork(DIMENSION_IN, DIMENSION_OUT)
-- trainSeveralLossesNNGraph(m, imgs, states,  NB_EPOCHS, LR, WEIGHT_IMPORTANCE_FOR_FIRST_INPUT)
