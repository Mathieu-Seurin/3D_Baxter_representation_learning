require 'nn'
require 'nngraph'


require 'cudnn'
require 'cunn'
require 'cutorch'

require 'functions'

DIMENSION_IN = 2 --Default for mobileRobot
DIMENSION_OUT = 2-- TODO change to 33 --Default for mobileRobot
LR = 0.05
NB_EPOCHS = 2--10 --1000
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
--
-- function sequentialTorchNNExample()
--    mlp= nn.Sequential();       --Create a network that takes a Tensor as input
--    mlp:add(nn.SplitTable(2))
--    c=nn.ParallelTable()      --The two Tensors go through two different Linear
--    c:add(nn.Linear(10,3))	   --Layers in Parallel
--    c:add(nn.Linear(10,7))
--    mlp:add(c)                 --Outputing a table with 2 elements
--    p=nn.ParallelTable()      --These tables go through two more linear layers
--    p:add(nn.Linear(3,2))	   -- separately.
--    p:add(nn.Linear(7,1))
--    mlp:add(p)
--    mlp:add(nn.JoinTable(1))   --Finally, the tables are joined together and output.
--
--    pred=mlp:forward(torch.randn(10,2))
--    print("prediction: ")
--    print(pred)
--
--    for i=1,NB_EPOCHS do             -- A few steps of training such a network..
--       x= torch.ones(10,2);
--       y= torch.Tensor(3); y:copy(x:select(2,1,1):narrow(1,1,3))
--       pred= mlp:forward(x)
--
--       criterion= nn.MSECriterion()
--       --FORWARD takes two params:  if we pass torch.randn(2,5), That input goes
--       -- into each of the nn.Linear(3,5) and nn.Linear(5,5) defined inputs, If you pass torch.randn(2, 5),
--       -- it is a vector of size 5 (with mini-batch size 2)
--       local err = criterion:forward(pred, y) --
--       local gradients = criterion:backward(pred, y);
--     --   print('x and y: ')
--     --   print(x,y)
--       mlp:zeroGradParameters();
--       mlp:backward(x, gradients);
--       mlp:updateParameters(LR);  --exclusive of nn?
--       print('Error (MSE):')
--       print(err)
--    end
-- end

function printAll(t, name)
    print(name)
    print(t)
    print (type(t)); print(#t)
end

---------------------
--Torch.nngraph Version Network
-- state_out_dim is a finetunable param. Initially, as in Jonchowscki, 2
--() converts nn.module to nngraph.node
---------------------
function stateAndAction2stateAndRewardModule(state_in_dim, state_out_dim)
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
    local inStateT = nn.Identity()()
    local inAction = nn.Identity()()
    -- print('inStateT must be an instance of nngraph.node!: ')
    -- print(inStateT)
    --hidd_state = nn.Linear(state_in_dim, state_out_dim)(inStateT)
    local stateTAndAction = nn.JoinTable(2)({inStateT, inAction}) --concat along the colums, one column next to other (1 would concatenate along rows, and means always the batch size input to the network)
    --nn.Concat(2)() -- Use only nn.ConcatTable() for nngraph architectures, do not mix torch.nngraph and torch.nn!-- nn.Concat(1) -- dim throuh which inputs are concatenated: inAction, stateT)
    --stateAndAction:add(inStateT)
    --stateAndAction:add(inAction) -- Action does not need to be encoded nor perturbed, it needs to help predict the reward
    --hidd_action = nn.Linear(state_in_dim, state_out_dim)(inAction)
    -- B)
    -- inStateT = nn.Linear(state_in_dim, state_out_dim)() -- Linear layer with input size 1 and output size 2
    -- inAction = nn.Linear(state_in_dim, state_out_dim)()
    -- outStateT1 = nn.ReLU()(nn.Linear(state_out_dim, state_out_dim)(inStateT)) -- equivalent to applying left to right or what in Python would be Linear(Tanh(h1))
    -- outRewardT = nn.HardShrink(0)(nn.Linear(state_out_dim, state_in_dim)(inAction)) --NOTE Reward T or T+1?
    -- --stateAndAction = nn.Concat()({hh1, hh2}) --Combining them by  CONCAT? before or after the activ function?
    -- print('stateTAndAction concatenated:')
    -- print(stateTAndAction)
    local outStateT1 = nn.Linear(state_out_dim, state_out_dim)(stateTAndAction)-- equivalent to applying left to right or what in Python would be Linear(Tanh(h1))
    --outRewardT = nn.Linear(state_out_dim, 1)(hidd_state) --NOTE Reward T or T+1? (in any case, 1 output numeric value)
    --option B: --Predict reward based on direct state or encoded hidd_state?
    local outRewardT = nn.Linear(state_in_dim, 1)(inStateT)
    --it in a 2 tensor input 2 tensor output network to create a bottle neck -- nn.Concat concatenates the output of one layer of "parallel" modules along the provided dimension"
    --outState = nn.ReLU()(stateAndAction) --nn.Sigmoid()(stateAndAction)
    --No need for any activ. function at all because our states should not be constrained in any range outReward = nn.HardShrink(0)(stateAndAction)
    --For rewards of -1, 0 and 10 ---TODO: add xavier weight initialization? https://github.com/Mathieu-Seurin/baxter_representation_learning_3D/blob/master/models/topUniqueSimplerWOTanh.lua
    print('outStateT1:')
    print(outStateT1)
    print('outRewardT:')
    print(outRewardT)
    --local stateT1AndReward = nn.JoinTable(2)({outStateT1, outRewardT})
    local gmod = nn.gModule({inStateT, inAction}, {outStateT1. outRewardT}) -- Parameters are Input and Output to our network  --Alterhative to nn.JoinTable?
    printNetworkGraph(gmod)
    return gmod
end


----------------
------TORCH.NNGraph VERSION
----------------
function train(gmod, input, output, epochs, LR)
    local MSECrit = nn.MSECriterion() --.cuda()
    MSECrit.sizeAverage = false --to avoid division by 0
    print('train: input, output and sizes: ',input, output, #input,#output)
    local loss = torch.Tensor(epochs):fill(0) -- training loss initialization
    for i=1, epochs do
        print('updating Gradient with NNGraph... epoch ',i)        -- A few steps of training such a network..
        loss[i] = updateGradient(gmod, input, output, MSECrit, LR)--{input1, input2}, {output1, output2}, multiCrit, LR)
        print('Error: (MSE loss):', loss[i])
        -- gmod:updateOutput(input)
        -- gmod:updateGradInput(input, output)  -- takes x, dx as input
        -- gmod:accGradParameters(input, output)
    end
    --local multiCrit = nn.MultiCriterion():add(lossCrit1, weightImportanceForFirstInput):add(lossCrit2, 1-weightForFirstInput) -- avged sum of criteria
    gmod:evaluate() --Should be done only once per whole training, needed really only if doing dropout
    return loss
end

----------------
------TORCH.NN AND TORCH.NNGRAPH? VERSION
--Uses a typical generic gradient update function
----------------
function updateGradient(net, x, y, criterion, learningRate)
    --printAll(x,'x')
    local y_pred = net:forward(x) --network forward step
    print("prediction: ", y_pred)
    local loss = criterion:forward(y_pred, y) -- -- network error evaluation (output, target)
    local gradients = criterion:backward(y_pred, y) --loss gradients
    net:zeroGradParameters() -- zero reset of gradients,  --sets them to zero, needs to be done in each fwd and backward update, otherwise Torch internally accumulates the sum of all gradients
    net:backward(x, gradients) --network backward step  (outputs, target)
    net:updateParameters(learningRate) --given the LR, update parameters
    return loss
end


-------------------------------------------------------------------------------
--TESTS
--First NeuralNetworksForBeginners
local statesT = torch.rand(2,2) --torch.Tensor{{0,1},{1,1}}--, {1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}}
--clockwise from 12 o'clock to 10.30, where the reward finally is
local actions = torch.rand(2,2) --torch.Tensor{{0,1},{1,1}}--, {1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}}
local statesT1 = torch.rand(2,2) --torch.Tensor{{1,1}, {1,0}}--,{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1}}
--local rewards = torch.Tensor{0,0,0,0,0,0,0,1} --torch.rand(10)
local rewards = torch.rand(2,1) --torch.Tensor(2,1)-- torch.rand(2)--{torch.rand(1), torch.rand(1)}

--
print('Building network..')
local n = stateAndAction2stateAndRewardModule(DIMENSION_IN, DIMENSION_OUT)
print('Network 1:')
print(n)
local finalLoss = train(n, {statesT, actions}, {statesT1, rewards}, NB_EPOCHS, LR)
