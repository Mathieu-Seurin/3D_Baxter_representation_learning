require 'nn'
require 'nngraph'


---INVERSE MODEL: : Given state_t and state_t+1, predict the action (needed to reach that state).
--  This is common problem in planning and navigation where we have the goal we want
-- to reach and we need to find the actions that take us there.


--TODO: difference with INVERSE RL? see https://stats.stackexchange.com/questions/189067/how-to-make-a-reward-function-in-reinforcement-learning
--TODO: reward prediction model (ICM where given a_t, s_t and s_t+1, we predict reward t)
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


BATCH_SIZE = 2
DIMENSION_ACTION = 2
DIMENSION_IN = 2
DIMENSION_OUT = DIMENSION_ACTION  --TODO: test that it also works when input and output dimensions are different
NUM_CLASS = 3 --3 DIFFERENTS REWARDS

NUM_HIDDEN_UNITS = 5 --TODO: what is ideal size? see ICM inverse model and forward model of loss is its own reward.



local M = {}

function getModel(dimension_out)

   local state_t0 = nn.Identity()()
   local state_t1 = nn.Identity()()

   state_and_next_state = nn.JoinTable(2)({state_t0, state_t1})

   action_prediction = nn.Linear(NUM_HIDDEN_UNITS, dimension_out)(nn.Linear(DIMENSION_IN *2, NUM_HIDDEN_UNITS)(state_and_next_state))

   g = nn.gModule({state_t0, state_t1}, {action_prediction})
   return g
end


function train_model(model_graph)
    --https://github.com/torch/nn/blob/master/doc/criterion.md#nn.CrossEntropyCriterion
    --Basically, you don't put any activation unit at the end of you network
    -- this criterion calculate the logsoftmax and the classification loss
    -- WHAT SHOULD BE THE CRITERION LOSS FUNCTION IN AN INVERSE MODEL?
    local crit = nn.MSECriterion() --    local crit2 = nn.MSECriterion()
    batch_state_t = torch.randn(BATCH_SIZE, DIMENSION_IN) --Returns a Tensor filled with random numbers from a normal distribution with zero mean and variance of one.
    batch_state_t1 = torch.randn(BATCH_SIZE, DIMENSION_IN)

    batch_action = torch.randn(BATCH_SIZE, DIMENSION_ACTION) -- print(batch_state_t1)--[torch.DoubleTensor of size 2x2]

    -- Takes an input object, and computes the corresponding output of the module.
    -- In general input and output are Tensors. However, some special sub-classes like table layers might expect something else.
    -- After a forward(), the returned output state variable should have been updated to the new value.
    output_action_var = model_graph:forward({batch_state_t, batch_state_t1})  --    print(output_state_var)
    --NOTE WE NEED TO DO A FWD AND BACKWARD PASS PER LOSS FUNCTION (CRITERION) WE ARE USING:
    local loss1 = crit:forward(output_action_var, batch_action)
    --loss2 = crit:forward(output_action_var[1], batch_state_t1)
    print('losses for criterion : '..loss1)--.." "..loss2)

    local grad1 = crit:backward(output_action_var, batch_action)
    --grad2 = crit1:backward(output_action_var[1], batch_state_t1)
    print('gradients for criterion : ')
    print(grad1)
    --print(grad2)

    --[gradInput] backward(input, gradOutput) Performs a backpropagation step through the module, with respect to the
    -- given input. In general this method makes the assumption forward(input) has been called before, with the same
    --input. This is necessary for optimization reasons. If you do not respect this rule, backward() will compute
    --incorrect gradients.
    --In general input and gradOutput and gradInput are Tensors. However, some special sub-classes like table layers might expect something else. Please, refer to each module specification for further information.
    --A backpropagation step consist in computing two kind of gradients at input given gradOutput (gradients with respect to the output of the module). This function simply performs this task using two function calls:
    -- A function call to updateGradInput(input, gradOutput).
    -- A function call to accGradParameters(input,gradOutput,scale).
    res = model_graph:backward({batch_state_t, batch_state_t1}, grad1)
    print('result from backward pass:  ')
    print(res)
end


g = getModel(DIMENSION_OUT)
train_model(g)

-- M.getModel = getModel(DIMENSION_OUT)
-- return M
