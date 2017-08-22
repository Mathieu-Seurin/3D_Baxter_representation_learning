require 'nn'
require 'nngraph'


--FORWARD MODEL (OR PREDICTIVE MODEL):
--Given state and action taken in time t, predict next state at t+1 that that action will lead us to.
-- I.e., "predicts" the future state


--  test_nn_graph.lua is an example of how nngraph should be done. Also a note:
--  never use updateOutput and updateGradInput, Only use forward and backward.
--  Basically, forward calls updateOutput + other stuff to retain the gradients etc.
--  And backward calls updateGradInput + other stuff to retain gradients etc. In conclusion,
--  it's better to call forward/backward because some models are doing more than just calling updateOutput etc.
-- SEE DOCUMENTATION IN https://github.com/torch/nn/blob/master/doc/module.md
--https://groups.google.com/forum/#!topic/torch7/SzC84-7HZsU


BATCH_SIZE = 8--2
DIMENSION_IN = 3
DIMENSION_OUT = 3 --DIMENSION_IN -- state in at time t should be same dim that state at time t+1
DIMENSION_ACTION = 2
NUM_CLASS = 3 --3 DIFFERENTS REWARDS

NUM_HIDDEN_UNITS = 5 
--ICM config:
--The forward model is constructed by concatenating φ(s_t) with a_t and passing it into a sequence of two fully connected layers with 256 and 288 units respectively
FC_UNITS_LAYER1 = 256
FC_UNITS_LAYER2 = 288
ETA = 1-- ? hyperparam, scaling factor for reward prediction (based on next state prediction loss)

local M = {}

function getModel(dimension_out)
    -- TODO: REAL FORWARD MODEL IN ICM PAPER ONLY PREDICTS NEXT STATE< NOT ALSO THE REWARD.
    --SIMPLIFY? or add more past states in order to predict reward? (see UNREAL Jaderberg paper)
   local state_t0 = nn.Identity()() -- notice, if not local vars, it can take the value from other models in the current folder and run as normal! that is LUA!
   local a0 = nn.Identity()()

   state_and_action = nn.JoinTable(2)({state_t0, a0})

   --Read pipeline application from left to right:
   reward_prediction = nn.Linear(NUM_HIDDEN_UNITS, NUM_CLASS)(nn.Linear(DIMENSION_IN, NUM_HIDDEN_UNITS)(state_t0))
   state_t1 = nn.Linear(NUM_HIDDEN_UNITS, dimension_out)(nn.Linear(DIMENSION_IN + DIMENSION_ACTION, NUM_HIDDEN_UNITS)(state_and_action))

   --TODO: m = nn.SpatialConvolution(1,3,2,2) -- learn 3 2x2 kernels --print(m.weight) -- initially, the weights are randomly initialized
   g = nn.gModule({state_t0, a0}, {state_t1, reward_prediction})
   return g

      -- whole_net:add(pretrain_net)
      -- whole_net:add(nn.Linear(512,Dimension))
      -- whole_net:evaluate()
      -- return whole_net
end

--Once you have multiple outputs in your network (by using for example
--ConcatTable), to compute different losses for different outputs you can use ParallelCriterion  TODO: check it works the same
--https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ParallelCriterion
--Or you can separately compute the loss and gradOutput of each output, then add them together for further use. It also works.
function train_model(model_graph)
    --https://github.com/torch/nn/blob/master/doc/criterion.md#nn.CrossEntropyCriterion
    --Basically, you don't put any activation unit at the end of you network
    -- this criterion calculate the logsoftmax and the classification loss. CrossEntropyCriterion behaves similarly to
    -- ClassNLLCriterion: the input should be batch_size x number_of_classes, but the
    --target will be only batch_size, with elements in the range 1, 2, ..., number_of_classes.
    --However, if your output and target are both n*n image tensors ？In this case，you can use cudnn.SpatialClassNLLCriterion  --TODO for general model with input images
    --TODO: modify loss and gradient of reward as in ICM for it to be the difference btw real and estimated state?
    --The reward is computed using a scaling factor to generate the curiosity based intrinsic reward signal,
    local crit1 = nn.MSECriterion() -- in ICM, only this one is used *1/2 and for reward prediction, the discrepancy on next state prediction is used TODO: future
    local crit2 = nn.CrossEntropyCriterion() --TODO: use weights param for umbalanced dataset (sparse rewards)
    --Due to the behaviour of the backend code, it is necessary to set sizeAverage to false when calculating losses in non-batch mode.
    crit2.nll.sizeAverage = false --The division by n can be avoided
    --local critJoint = nn.ParallelCriterion():add(crit1, 0.5):add(crit2) -- Needed? does not allow easy weighting with ETA

    batch_state = torch.randn(BATCH_SIZE, DIMENSION_IN) --Returns a Tensor filled with random numbers from a normal distribution with zero mean and variance of one.
    batch_action = torch.randn(BATCH_SIZE, DIMENSION_ACTION)

    batch_state_t1 = torch.randn(BATCH_SIZE, DIMENSION_OUT) -- print(batch_state_t1)--[torch.DoubleTensor of size 2x2]
    batch_rew = torch.ones(BATCH_SIZE)--, NUM_CLASS) --[torch.DoubleTensor of size 2]
    --or should it be torch.IntTensor{1,8} as used by ClassNLLCriterion?
    -- Takes an input object, and computes the corresponding output of the module.
    -- In general input and output are Tensors. However, some special sub-classes like table layers might expect something else.
    -- After a forward(), the returned output state variable should have been updated to the new value.
    local output_state_var = model_graph:forward({batch_state, batch_action})
    print('output state var ')
    print(output_state_var)
    --NOTE WE NEED TO DO A FWD AND BACKWARD PASS PER LOSS FUNCTION (CRITERION) WE ARE USING:
    local loss1 = crit1:forward(output_state_var[1], batch_state_t1) --()predicted and GT as params)
    local loss2 = crit2:forward(output_state_var[2], batch_rew) * ETA  --
    --local lossJoint = critJoint:forward({batch_state, batch_action}, {batch_state_t1, batch_rew})

    print('losses for criterion 1 and 2: '..loss1..' '..loss2)--..' '..lossJoint)

    local grad1 = crit1:backward(output_state_var[1], batch_state_t1)
    local grad2 = crit2:backward(output_state_var[2], batch_rew)
    --local gradJoint = critJoint:backward(output_state_var, {batch_state_t1, batch_rew})

    print('gradients for criterion 1 and 2, and joint: ')
    print(grad1)
    print(grad2)
    --print(gradJoint)

    --[gradInput] backward(input, gradOutput) Performs a backpropagation step through the module, with respect to the
    -- given input. In general this method makes the assumption forward(input) has been called before, with the same
    --input. This is necessary for optimization reasons. If you do not respect this rule, backward() will compute
    --incorrect gradients.
    --In general input and gradOutput and gradInput are Tensors. However, some special sub-classes like table layers might expect something else. Please, refer to each module specification for further information.
    --A backpropagation step consist in computing two kind of gradients at input given gradOutput (gradients with respect to the output of the module). This function simply performs this task using two function calls:
    -- A function call to updateGradInput(input, gradOutput).
    -- A function call to accGradParameters(input,gradOutput,scale).

    --model_graph:zeroGradParameters() -- zero the internal gradient buffers of the network

    res = model_graph:backward({batch_state, batch_action},{grad1,grad2})
    print('result from backward pass:  ')
    print(res)
    print(res[1])
    print(res[2])
    --resJoint = model_graph:backward({batch_state, batch_action}, {gradJoint})
    -- print('joint result: ')
    -- print(resJoint)
end


local g = getModel(DIMENSION_OUT)
train_model(g)
--
-- M.getModel = getModel(DIMENSION_OUT)
-- return M
