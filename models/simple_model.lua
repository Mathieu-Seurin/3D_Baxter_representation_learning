require 'nn'
require 'nngraph'
--TODO for visualization: generateGraph = require 'optnet.graphgen'  https://github.com/fmassa/optimize-net   https://discuss.pytorch.org/t/print-autograd-graph/692/24

--require 'functions'

---INVERSE MODEL: : Given state_t and state_t+1, predict the action (needed to reach that state).
--  This is common problem in planning and navigation where we have the goal we want
-- to reach and we need to find the actions that take us there.


--TODO: difference with INVERSE RL? see https://stats.stackexchange.com/questions/189067/how-to-make-a-reward-function-in-reinforcement-learning
--TODO: reward prediction model (ICM where given a_t, s_t and s_t+1, we predict reward t),
-- see pred_bonus in https://github.com/pathak22/noreward-rl/blob/master/src/model.py
---TODO Augment with full ICM using  Intrinsic reward signal is computed as the difference between the next state and the estimated next state, pondered by a η/2 (η > 0)  scaling factor
--inverse RL or "apprenticeship learning", which generates a reward function that would reproduce observed behaviours.
--  Finding the best reward function to reproduce a set of observations can also be
-- implemented by MLE, bayesian, or information theoretic methods.  finding the reward function is the hardest part of the problem, it is intimately tied up with how you specify the state space. For example in a time-dependent problem, the distance to the goal often makes a poor reward function (e.g. in the mountain car problem).
--Such situations can be solved by using higher dimensional state spaces (hidden states or memory traces), or by hierarchical RL.

--  test_nn_graph.lua is an example of how nngraph should be done. Also a note:
--  never use updateOutput and updateGradInput, Only use forward and backward.
--  Basically, forward calls updateOutput + other stuff to retain the gradients etc.
--  And backward calls updateGradInput + other stuff to retain gradients etc. In conclusion,
--  it's better to call forward/backward because some models are doing more than just calling updateOutput etc.
-- SEE DOCUMENTATION IN https://github.com/torch/nn/blob/master/doc/module.md

--Model inspired from ICM:
-- The intrinsic curiosity module consists of the forward and the inverse
-- model. The inverse model first maps the input state
-- (st) into a feature vector φ(st) using a series of four convolution
-- layers, each with 32 filters, kernel size 3x3, stride
-- of 2 and padding of 1. ELU non-linearity is used after
-- each convolution layer. The dimensionality of φ(st) (i.e.
-- the output of the fourth convolution layer) is 288. For the
-- inverse model, φ(st) and φ(st+1) are concatenated into a
-- single feature vector and passed as inputs into a fully connected
-- layer of 256 units followed by an output fully connected
-- layer with 4 units to predict one of the four possible
-- actions. The forward model is constructed by concatenating
-- φ(st) with at and passing it into a sequence of two fully
-- connected layers with 256 and 288 units respectively. The
-- value of β is 0.2, and λ is 0.1. The Equation (7) is minimized
-- with learning rate of 1e − 3.

BATCH_SIZE = 4
DIMENSION_ACTION = 2
DIMENSION_IN = 512
DIMENSION_OUT = DIMENSION_ACTION
NUM_CLASS = 3 --3 DIFFERENTS REWARDS

--FROM ICM:
HIDDEN_UNITS = 5 --TODO: what is ideal size? see ICM inverse model and forward model of loss is its own reward.
FC_UNITS_LAYER1 = 256
FC_UNITS_LAYER2 = 4
--TODO add ELU after each conv layer  and  four convolution layers, each with
NB_FILTERS = 32
KERNEL_SIZE = 3 --3x3
STRIDE = 2
PADDING = 1

local M = {}


function saveNetworkGraph(gmodule, title, show)
    -- gmod is what we send to forward and backward pass
    if not show then
        graph.dot(gmodule.fg, title ) --'Forward Graph')
    else --show and save          --filename = '../modelGraphs/fwdGraph'..title --Using / in path gives segmentation fault error!?
        print('saving network graph in main project dir: '..title)
        --if not file_exists(filename) then
        graph.dot(gmodule.fg, title, title)--, filename) --TODO fix: gives Graphviz Segmentation fault (core dumped)
    end
end

function getSimpleFeatureEncoderNetwork(dimension_out)
    --Input: Image
    --Output: state
    local img = nn.Identity()()
    state_prediction = nn.ReLU()(nn.Linear(DIMENSION_IN, dimension_out)(img))
    --state_prediction = nn.Linear(HIDDEN_UNITS, dimension_out)(state_prediction) --NEEDED?
    g = nn.gModule({img}, {state_prediction})

    -- Initialisation : "Understanding the difficulty of training deep feedforward neural networks"
    local g = require('weight-init')(g, 'xavier') --    print('Simple Net\n' .. g:__tostring());
    saveNetworkGraph(g, 'SimpleFeatureEncoderNetwork', true)
    return g
end

function train_model(model_graph)
    --https://github.com/torch/nn/blob/master/doc/criterion.md#nn.CrossEntropyCriterion
    --Basically, you don't put any activation unit at the end of you network
    -- this criterion calculate the logsoftmax and the classification loss
    -- WHAT SHOULD BE THE CRITERION LOSS FUNCTION IN AN INVERSE MODEL? If we had discrete actions, as in ICM, a soft-max distribution accross all possible actions (amounts to MLE of theta under a multinomial distribution)
    local crit = nn.MSECriterion() --    local crit2 = nn.MSECriterion()
    batch_img = torch.randn(BATCH_SIZE, DIMENSION_IN) --Returns a Tensor filled with random numbers from a normal distribution with zero mean and variance of one.
    batch_state = torch.randn(BATCH_SIZE, DIMENSION_OUT) -- print(batch_state_t1)--[torch.DoubleTensor of size 2x2]
    model_graph:zeroGradParameters() -- zero the internal gradient buffers of the network

    -- Takes an input object, and computes the corresponding output of the module.
    -- In general input and output are Tensors. However, some special sub-classes like table layers might expect something else.
    -- After a forward(), the returned output state variable should have been updated to the new value.
    local output_var = model_graph:forward(batch_img)
    print(' output_var  ');  print(output_var)
    local loss = crit:forward(output_var, batch_state)
    print('loss for MSE criterion : '..loss)

    local grad = crit:backward(output_var, batch_state)
    print('gradients for criterion : ')
    print(grad)

    --[gradInput] backward(input, gradOutput) Performs a backpropagation step through the module, with respect to the
    -- given input. In general this method makes the assumption forward(input) has been called before, with the same
    --input. This is necessary for optimization reasons. If you do not respect this rule, backward() will compute
    --incorrect gradients.
    --In general input and gradOutput and gradInput are Tensors. However, some special sub-classes like table layers might expect something else. Please, refer to each module specification for further information.
    --A backpropagation step consist in computing two kind of gradients at input given gradOutput (gradients with respect to the output of the module). This function simply performs this task using two function calls:
    -- A function call to updateGradInput(input, gradOutput).
    -- A function call to accGradParameters(input,gradOutput,scale).
    res = model_graph:backward({batch_img}, grad)
    --print('result from backward pass:  '); print(res)
end



local g = getSimpleFeatureEncoderNetwork(DIMENSION_OUT)
train_model(g)
