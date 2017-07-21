require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'nngraph'
require 'MSDC'
require 'functions'
require 'printing'
require "Get_Images_Set"
require 'optim_priors'
require 'definition_priors'

-- THIS IS WHERE ALL THE CONSTANTS SHOULD COME FROM
-- See const.lua file for more details
require 'const'
-- try to avoid global variable as much as possible

function Rico_Training(Models,priors_used)
   local rep_criterion=get_Rep_criterion()
   local prop_criterion=get_Prop_criterion()
   local caus_criterion=get_Caus_criterion()
   local temp_criterion=nn.MSDCriterion()

   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
      local loss_rep, loss_caus, loss_prop, loss_temp, loss_close, loss_fix = 0, 0, 0, 0, 0, 0
      -- just in case:
      collectgarbage()

      local batch, action1, action2
      -- get new parameters
      if x ~= parameters then
         parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      --===========
      local mode='Temp' --Same for continuous or discrete actions
      if applying_prior(priors_used, mode) then
          batch=getRandomBatchFromSeparateList(BATCH_SIZE,mode)
          loss_temp, grad=doStuff_temp(Models,temp_criterion, batch,COEF_TEMP)
          TOTAL_LOSS_TEMP = loss_temp + TOTAL_LOSS_TEMP
      end

      mode='Prop'
      if applying_prior(priors_used, mode) then
          batch, action1, action2 = getRandomBatchFromSeparateList(BATCH_SIZE,mode)
          loss_prop, gradProp=doStuff_Prop(Models,prop_criterion,batch,COEF_PROP, action1, action2)
          TOTAL_LOSS_PROP = loss_prop + TOTAL_LOSS_PROP
      end

      --==========
      mode='Caus'  --Not applied for BABBLING data (sparse rewards)
      if applying_prior(priors_used, mode) then
        batch, action1, action2 = getRandomBatchFromSeparateList(BATCH_SIZE,mode)
        loss_caus, gradCaus=doStuff_Caus(Models,caus_criterion,batch,COEF_CAUS, action1, action2)
        TOTAL_LOSS_CAUS = loss_caus + TOTAL_LOSS_CAUS
      end

      --==========
      mode='Rep'
      if applying_prior(priors_used, mode) then
          batch, action1, action2 = getRandomBatchFromSeparateList(BATCH_SIZE,mode)
          loss_rep, gradRep=doStuff_Rep(Models,rep_criterion,batch,COEF_REP, action1, action2)
          TOTAL_LOSS_REP = loss_rep + TOTAL_LOSS_REP
      end

      mode='make_reward_closer'
      if applying_prior(priors_used, mode) then
          batch = getRandomBatchFromSeparateList(BATCH_SIZE,mode)
          loss_reward_closer, gradClose=doStuff_temp(Models,temp_criterion,batch,COEF_CLOSE) --Just minimizing mse criterion, so we can use temp criterion
          TOTAL_LOSS_CLOSE = loss_reward_closer + TOTAL_LOSS_CLOSE
      end

      mode='fixed_point'
      if applying_prior(priors_used, mode) then
          batch = getRandomBatchFromSeparateList(BATCH_SIZE,mode)
          loss_fix, gradClose=doStuff_temp(Models,temp_criterion,batch,COEF_FIX) --Just minimizing mse criterion, so we can use temp criterion
          TOTAL_LOSS_FIX = loss_fix + TOTAL_LOSS_FIX
      end

      --NOTE: shouldnt gradParameters be here  the sum of all gradRep, gradCaus, etc?
      --Grad parameters is a tensor containing the internal gradient of all model's parameters
      -- So the sum of gradients is already present in there
      return loss_rep+loss_caus+loss_prop+loss_temp, gradParameters
    end

    --sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }
    --parameters, loss=optim.sgd(feval, parameters, sgdState)
    optimState={learningRate=LR, learningRateDecay=LR_DECAY}

    if SGD_METHOD == 'adagrad' then
        parameters,loss = optim.adagrad(feval,parameters,optimState)
    elseif SGD_METHOD == 'adam' then
        parameters,loss = optim.adam(feval,parameters,optimState)
    else
       parameters,loss = optim.adamax(feval,parameters,optimState)
    end

    -- loss[1] table of one value transformed in just a value
    -- grad[1] we use just the first gradient to print the figure (there are 2 or 4 gradient normally)
    return loss[1], grad
end

function train(Models, priors_used)

   LOG_SEQ_USED = {}
   
    local NB_BATCHES= math.ceil(NB_SEQUENCES*AVG_FRAMES_PER_RECORD/BATCH_SIZE/(4+4+2+2))
    --AVG_FRAMES_PER_RECORD to get an idea of the total number of images
    --div by 12 because the network sees 12 images per iteration (i.e. record)
    -- (4*2 for rep and prop +  2*2 for temp and caus = 12)
    print(NB_SEQUENCES..' : sequences. '..NB_BATCHES..' batches')
    print("Number of epochs : ", NB_EPOCHS)
    for epoch=1, NB_EPOCHS do
       print('--------------Epoch : '..epoch..' ---------------')

       TOTAL_LOSS_TEMP,TOTAL_LOSS_CAUS,TOTAL_LOSS_PROP, TOTAL_LOSS_REP, TOTAL_LOSS_CLOSE, TOTAL_LOSS_FIX = 0,0,0,0,0,0

       xlua.progress(0, NB_BATCHES)
       for numBatch=1, NB_BATCHES do
          Loss, Grad = Rico_Training(Models,priors_used)
          xlua.progress(numBatch, NB_BATCHES)

       end

       print("Loss Temp", TOTAL_LOSS_TEMP/NB_BATCHES/BATCH_SIZE)
       print("Loss Prop", TOTAL_LOSS_PROP/NB_BATCHES/BATCH_SIZE)
       print("Loss Caus", TOTAL_LOSS_CAUS/NB_BATCHES/BATCH_SIZE)
       print("Loss Rep", TOTAL_LOSS_REP/NB_BATCHES/BATCH_SIZE)

       if BRING_CLOSER_REWARD then
          print("Loss Close", TOTAL_LOSS_CLOSE/NB_BATCHES/BATCH_SIZE)
       end

       if BRING_CLOSER_REF_POINT then
          print("Loss Fix", TOTAL_LOSS_FIX/NB_BATCHES/BATCH_SIZE)
          --You don't need to see the log at every time step, the first 3 will do
          if epoch == 1 or epoch ==2 or epoch==3 then
             print("Log_Seq",LOG_SEQ_USED)
          end

       end


       save_model(Models.Model1, NAME_SAVE) --TODO Do we need to write NB_EPOCH TIMES? isnt enough the last time to write once and not overwrite NB_EPOCH TIMES?
   end
   log_model_params()
   return Models.Model1, NAME_SAVE
end



local function main(params)
    print("\n\n>> script.lua: main model builder")
    set_hyperparams(params)--    print('In DATA_FOLDER: '..DATA_FOLDER..' params: ')
    print(params)
    print_hyperparameters()

        
    if USE_CUDA then
       require 'cunn'
       require 'cudnn'
    end

    local records_paths = Get_Folders(DATA_FOLDER, 'record') --local list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)
    NB_SEQUENCES= #records_paths

    if DATA_FOLDER == COMPLEX_DATA then
       NB_SEQUENCES = NB_SEQUENCES - 1 -- To avoid looking at the test set
    end

    if NB_SEQUENCES ==0  then --or not folder_exists(DATA_FOLDER) then
        error('Error: data was not found in input directory INPUT_DIR= '.. DATA_FOLDER)
    end

    if LOGGING_ACTIONS then
       print("LOGGING ACTIONS")
       LOG_ACTION = {}

       for i=1,NB_SEQUENCES do
          LOG_ACTION[#LOG_ACTION+1] = {}
       end
    end

    ALL_SEQ = precompute_all_seq()

    for nb_test=1, #PRIORS_CONFIGS_TO_APPLY do
       if RELOAD_MODEL then
          print("Reloading model in "..SAVED_MODEL_PATH)
          Model = torch.load(SAVED_MODEL_PATH):double()
       else
          print("Getting model in : "..MODEL_ARCHITECTURE_FILE)
          require(MODEL_ARCHITECTURE_FILE)
          Model=getModel(DIMENSION_OUT)
          --graph.dot(Model.fg, 'Our Model')
       end

       if USE_CUDA then
          Model=Model:cuda()
       end

       parameters,gradParameters = Model:getParameters()

       Model2=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
       Model3=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
       Model4=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
       Models={Model1=Model,Model2=Model2,Model3=Model3,Model4=Model4}

       local priors_used= PRIORS_CONFIGS_TO_APPLY[nb_test]
       local Log_Folder=Get_Folder_Name(LOG_FOLDER, priors_used)

       print("Experiment "..nb_test .." (Log_Folder="..Log_Folder.."): Training model using priors: ")
       print(priors_used)
       train(Models, priors_used)
       print_experiment_config()
    end

    if LOGGING_ACTIONS then
       print("LOG_ACTION")
       for key,items in ipairs(LOG_ACTION) do
          i = 0
          for k,j in pairs(items) do
             i = i+1
          end
          print(key,i)
       end
    end
end

local cmd = torch.CmdLine()
-- Basic options
cmd:option('-use_cuda', false, 'true to use GPU, false (default) for CPU only mode')
cmd:option('-use_continuous', false, 'true to use a continuous action space, false (default) for discrete one (0.5 range actions)')
cmd:option('-data_folder', STATIC_BUTTON_SIMPLEST, 'Possible Datasets to use: staticButtonSimplest, mobileRobot, staticButtonSimplest, simpleData3D, pushingButton3DAugmented, babbling')
cmd:option('-mcd', 0.5, 'Max. cosine distance allowed among actions for priors loss function evaluation (MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)')
cmd:option('-sigma', 0.1, "Sigma: denominator in continuous actions' extra factor (CONTINUOUS_ACTION_SIGMA)")
--TODO Set best mcd and sigma after grid search

local params = cmd:parse(arg)  --TODO function to get all command line arguments that are the same right now for all Lua scripts, only in one function.
main(params)
