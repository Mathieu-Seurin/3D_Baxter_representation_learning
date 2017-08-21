require 'nn'
require 'nngraph'

--  test_nn_graph.lua is an example of how nngraph should be done. Also a note:
--  never use updateOutput and updateGradInput, Only use forward and backward.
--  Basically, forward calls updateOutput + other stuff to retain the gradients etc.
--  And backward calls updateGradInput + other stuff to retain gradients etc. In conclusion,
--  it's better to call forward/backward because some models are doing more than just calling updateOutput etc.


function define_model()

   state_t0 = nn.Identity()()
   a0 = nn.Identity()()

   state_and_action = nn.JoinTable(2)({state_t0, a0})

   reward_prediction = nn.Linear(NUM_HIDDEN_UNITS, NUM_CLASS)(nn.Linear(DIMENSION_IN, NUM_HIDDEN_UNITS)(state_t0))
   state_t1 = nn.Linear(NUM_HIDDEN_UNITS, DIMENSION_OUT)(nn.Linear(DIMENSION_IN + DIMENSION_ACTION, NUM_HIDDEN_UNITS)(state_and_action))

   g = nn.gModule({state_t0, a0}, {state_t1, reward_prediction})
   return g
end

BATCH_SIZE = 2
DIMENSION_IN = 3
DIMENSION_ACTION = 2
NUM_HIDDEN_UNITS = 5
DIMENSION_OUT = 2
NUM_CLASS = 3 --3 DIFFERENTS REWARDS


g = define_model()
crit1 = nn.MSECriterion()
crit2 = nn.CrossEntropyCriterion()

batch_state = torch.randn(BATCH_SIZE, DIMENSION_IN)
batch_action = torch.randn(BATCH_SIZE, DIMENSION_ACTION)

batch_state_t1 = torch.randn(BATCH_SIZE, DIMENSION_OUT)
batch_rew = torch.ones(BATCH_SIZE)

out = g:forward({batch_state,batch_action})

loss1 = crit1:forward(out[1], batch_state_t1)
loss2 = crit2:forward(out[2], batch_rew)

grad1 = crit1:backward(out[1], batch_state_t1)
grad2 = crit2:backward(out[2], batch_rew)

res = g:backward({batch_state,batch_action},{grad1,grad2})
