-- TODO change local var batchsize to global BATCH_SIZE
require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
--require 'cunn'
require 'nngraph'
require 'functions'
require 'printing'
require "Get_Images_Set"
require 'optim_priors'
require 'definition_priors'
require 'gnuplot'
require 'os'
require 'paths'

-- THIS IS WHERE ALL THE CONSTANTS SHOULD COME FROM
-- See const.lua file for more details
require 'const'
-- try to avoid global variable as much as possible

local cmd = torch.CmdLine()
-- Basic options
cmd:option('-use_cuda', false, 'true to use GPU version, false (default) for CPU only mode')
cmd:option('-use_continuous', false, 'true to use a continuous action space, false (default) for discrete one (0.5 range actions)')
cmd:option('-data_folder', STATIC_BUTTON_SIMPLEST, 'Possible Datasets to use: staticButtonSimplest, mobileRobot, staticButtonSimplest, simpleData3D, pushingButton3DAugmented, babbling')
cmd:option('-mcd', 0.4, 'Max. cosine distance allowed among actions for priors loss function evaluation (MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)')
cmd:option('-sigma', 0.6, "Sigma: denominator in continuous actions' extra factor (CONTINUOUS_ACTION_SIGMA)")

RELATIVE = false

local params = cmd:parse(arg)
set_hyperparams(params)
print("testing DATA_FOLDER", DATA_FOLDER)
local list_folders_images, list_txt_action,list_txt_button, list_txt_state, list_txt_posButton=Get_HeadCamera_View_Files(DATA_FOLDER)
NB_SEQUENCES = #list_folders_images

-- function for testing
-- indice: data sequnce
-- number: how many samples
function printSamples(Model, indice, number)
  data = load_seq_by_id(indice)
  for i = 1,number do
    local input = data.images[i]:double()
    local truth = getLabel(data, i)
    if USE_CUDA then
      input = input:reshape(1, input:size(1), input:size(2), input:size(3))
      input = input:cuda()
    end
    output = Model:forward(input)
    print('**** ---- ****')
    print('truth')
    print(truth)
    print('output')
    print(output)
    -- print(input[{1,1,{}}])
  end
end

---------------- model & criterion ---------------------------
-- TODO load modle using minimalNetModel (now )
require(MODEL_ARCHITECTURE_FILE) -- minimalnetmodel
-- Model = getModel(DIMENSION_IN)
if USE_CUDA then
  criterion = nn.MSECriterion():cuda()
else
  criterion = nn.MSECriterion()
end

function reinitNet()
  -- reinit weights for cross-validation
  local method = 'xavier'
  Model:clearState()
  Model:reset()
  Model = require('weight-init')(Model, method)
end

----- adding the case of relative postioning -------
function getLabel(data, index)
  -- get label of image i in data sequence
  local label = torch.Tensor(DIMENSION_IN)
  for i = 1, DIMENSION_IN do
    label[i] = data.Infos[i][index]
  end
  -- if RELATIVE then
    -- label = label - data.posButton-- TODO change data's sturecture to include posB
  -- end
  return label
end

-- simple evaluation of acurracy
function evaluate(Model, data)
  local n = (#data.images)
  local err = 0
  for i = 1,n do
    local input = data.images[i]:double()
    local truth = getLabel(data, i)
    if USE_CUDA then
      input = input:reshape(1, input:size(1), input:size(2), input:size(3))
      input = input:cuda()
      truth = truth:cuda()
    end
    output = Model:forward(input)
    -- err = err + (output - truth):pow(2):sum()
    err = err + criterion:forward(output, truth)
  end
  err = err / n
  return err
end

function train(Model, indice_val, verbose, final, evalTrain)
  -- For simpleData3D at the moment. Training using sequences 1-7, 8 as test.
  -- Given an indice_val, train and return the *errors* on training set as well
  -- as on validation set.
  -- Can be made general directly to mobileRobot?
  -- Test set has always sequnce id NB_SEQUENCES (last one)
  local batchSize = BATCH_SIZE
  local nb_epochs = NB_EPOCHS
  local lr = LR
  collectgarbage()
  Model:clearState()
  if RELATIVE then
    print("============ Using relative states =========")
  end
  local final = final or false
  local verbose = verbose or false
  local evalTrain = evalTrain or false -- output only evaluation on train set, as sanity check
  local nb_batches = math.ceil(NB_SEQUENCES * 90 / batchSize)
  local LRhalflife = 10

  local parameters, gradParameters = Model:getParameters()
  if verbose then
    print("============ Verbose mode =========")
    xlua.progress(0, nb_epochs)
    logger = optim.Logger('Log/' ..DATA_FOLDER..'Epoch'..nb_epochs..'Batch'..batchSize..'LR'..LR..'val'..indice_val..'.log')
    logger:setNames{'Validation Accuracy'}
    logger:display(false)
  end
  local optimState = {LearningRate = LR}

  -- Generate index, i = NB_SEQUENCES is the test set
  local index_train = {}
  for i = 1, NB_SEQUENCES - 1 do
    index_train[i] = i
  end
  if final then
    index_train[NB_SEQUENCES] = NB_SEQUENCES
    local indice_val = NB_SEQUENCES
  end
  table.remove(index_train, indice_val)

  if evalTrain then
    local indice_val = index_train[1]
  end

  local err_val = torch.Tensor(nb_epochs)
  local LRcount = 0
  for epoch = 1, nb_epochs do
    if LRcount == LRhalflife then
      lr = lr * 0.5
      LRcount = 0
      print("Learning rate reduced by half at epoch ", epoch)
    end
    LRcount = LRcount + 1
    optimState = {LearningRate = lr}
    for batch = 1, nb_batches do
      -- xlua.progress(0, nb_batches)
      -- load data sequence
      local indice = torch.random(1, #index_train)
      local data = load_seq_by_id(index_train[indice])
      assert(data, "Error loading data")

      local n = #data.Infos[1]
      local dim = #data.images[1]
      local batch = torch.Tensor(batchSize, dim[1], dim[2], dim[3])
      local labels = torch.Tensor(batchSize, DIMENSION_IN)
      -- get batches
      for k = 1, batchSize do
        i = torch.random(1,n)
        batch[k] = data.images[i]:double()
        labels[k] = getLabel(data,i)
      end
      -- closure for optim
      if USE_CUDA then
        batch = batch:cuda()
        labels = labels:cuda()
      end

      local feval = function(x)
          collectgarbage()
          if x ~= parameters then
              parameters:copy(x)
          end
          gradParameters:zero()
          local state = Model:forward(batch)
          local loss = criterion:forward(state,labels)
          local dloss_dstate = criterion:backward(state, labels)
          Model:backward(batch, dloss_dstate)
          return loss, gradParameters
      end
      -- xlua.progress(batch, nb_batches)
      optim.adam(feval, parameters, optimState) -- or adam
    end
    err_val[epoch] = evaluate(Model, load_seq_by_id(indice_val))
    if verbose then
      logger:add{err_val[epoch]}
      logger:style{'+-'}
      logger:plot()
      print(err_val[epoch])
      xlua.progress(epoch, nb_epochs)
    end
  end
  performance_val = err_val[nb_epochs]-- final error
  return performance_val, err_val
end

------- cross-validation ---------------------------
function cross_validation(K)
  -- K-fold cross-valition on epoch size, batch size, and learning rate
  -- need improving!
  local nb_epochSet = {30}
  local nb_batchSet = {16,32}
  -- lrSet = {0.01}
  local lrSet = {0.01,0.001}
  configs = {}
  nb_config = #nb_epochSet * #nb_batchSet *  #lrSet
  performances = torch.Tensor(nb_config)
  bestConfig = {}
  bestPerf = 100
  local count = 1
  print("iterating over configs")
  xlua.progress(0, nb_config)
  print("Using Model", MODEL_ARCHITECTURE_FILE)

  for i, nb_epochs in pairs(nb_epochSet) do
    for j, batchSize in pairs(nb_batchSet) do
      for k, lr in pairs(lrSet) do
        -- training, K-fold
        local avgPerf = 0
        configs[count] = {'Epoch = '..nb_epochs, 'Batch = '..batchSize, 'LR = '..lr}
        print(configs[count])
        for indice_val = 1, K do
          --  indice_val = torch.random(1, NB_SEQUENCES-1)
           print("validation sequence id: ", indice_val)
           local Model = getModel(DIMENSION_IN)
           if USE_CUDA then
             Model = Model:cuda()
           end
           avgPerf = avgPerf + train(Model, nb_epochs, batchSize, lr,
                                     indice_val, false, false, false)
        end
        performances[count] = avgPerf / K
        if performances[count] < bestPerf then -- save best config
          bestPerf = performances[count]
          bestConfig = {nb_epochs, batchSize, lr}
        end
        print(performances[count])
        count = count + 1
        xlua.progress(count - 1, nb_config)
      end
    end
  end
  -- print errors
  -- for i = 2, performances:size(2) do
  --   print("MSE", performances[i], configs[i])
  -- end
  -- pick the best model, apply on test set
  min, index = torch.min(performances, 1)
  print("best-model", configs[index[1]])
  local Model = getModel(DIMENSION_IN)
  if USE_CUDA then
    Model = Model:cuda()
  end
  train(Model, bestConfig[1], bestConfig[2], bestConfig[3], NB_SEQUENCES, true, true, false)
  print(evaluate(Model, load_seq_by_id(NB_SEQUENCES)))
  save_model(Model)
end

---------------- single run -----------------
function test_run(verbose)
  -- local LR = 0.001 --(hyper)parameters for training
  -- local nb_epochs = 50
  -- local batchSize = 16
  local err = torch.Tensor(NB_EPOCHS)
  -- local indice_val = NB_SEQUENCES
  local indice_val = NB_SEQUENCES
  print("Using Model", MODEL_ARCHITECTURE_FILE)
  local Model = getModel(DIMENSION_IN)
  if USE_CUDA then
    Model = Model:cuda()
  end
  _, err = train(Model, indice_val, verbose, true, false)
  print(evaluate(Model, load_seq_by_id(indice_val)))
  -- print("results from data seq "..indice_val.."with parameters (epoch, batch, lr)"..nb_epochs.." "..batchSize.." "..LR.." is")
  printSamples(Model, indice_val, 3)
  save_model(Model)
  ------ test if reinitiation works --------
  -- print(parameters:sum())
  -- reinitNet()
  -- print("after reinitiation")
  -- print(parameters:sum())
end

----------------- run ----------
-- cross_validation(1)

verbose = true
test_run(verbose)

--------- load model test---------
-- if file_exists('lastModel.txt') then
--   f = io.open('lastModel.txt', 'r')
--   path = f:read()
--   modelString = f:read()
--   print('MODEL: '..modelString)
--   f:close()
-- else
--   error("lastModel.txt not found")
-- end
-- local Model = torch.load(path..'/'..modelString):double()
-- printSamples(Model, 8, 3)
--------- load model test---------

----------------- codes for testing -----------------
-- data = load_seq_by_id(2)
-- print(data.images[1][{1,1,{}}])
-- print(#data.Infos[1])
-- print(data.posButton)
