require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'nngraph'

-- THIS IS WHERE ALL THE CONSTANTS SHOULD COME FROM
-- See const.lua file for more details
require 'const'
-- try to avoid global variable as much as possible

if USE_CUDA then
   require 'cunn'
end

require 'MSDC'
require 'functions'
require 'printing'
require "Get_Images_Set"
require 'optim_priors'
require 'definition_priors'

function Rico_Training(Models)

   local rep_criterion=get_Rep_criterion()
   local prop_criterion=get_Prop_criterion()
   local caus_criterion=get_Caus_criterion()
   local temp_criterion=nn.MSDCriterion()

   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
      -- just in case:
      collectgarbage()

      local action1, action2 --,lossTemp,lossProp,lossCaus,lossRep
      -- get new parameters
      if x ~= parameters then
         parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      --===========
      local mode='Temp' --Same for continuous or not
      local batch=getRandomBatchFromSeparateList(BATCH_SIZE,mode, USE_CONTINUOUS)
      LOSS_TEMP,grad=doStuff_temp(Models,temp_criterion, batch,COEF_TEMP)

      if USE_CONTINUOUS then
         --==========
         mode='Prop'
         batch, action1, action2 = getRandomBatchFromSeparateList(BATCH_SIZE,mode, USE_CONTINUOUS)
         LOSS_PROP,gradProp=doStuff_Prop_continuous(Models,prop_criterion,batch,COEF_PROP, action1, action2)

         --==========
         mode='Caus'
         batch, action1, action2 = getRandomBatchFromSeparateList(BATCH_SIZE,mode, USE_CONTINUOUS)
         LOSS_CAUS,gradCaus=doStuff_Caus_continuous(Models,caus_criterion,batch,COEF_CAUS, action1, action2)

         --==========
         mode='Rep'
         batch, action1, action2 = getRandomBatchFromSeparateList(BATCH_SIZE,mode, USE_CONTINUOUS)
         LOSS_REP,gradRep=doStuff_Rep_continuous(Models,rep_criterion,batch,COEF_REP, action1, action2)
      else
         --==========
         mode='Prop'
         batch = getRandomBatchFromSeparateList(BATCH_SIZE,mode, USE_CONTINUOUS)
         LOSS_PROP,gradProp=doStuff_Prop(Models,prop_criterion,batch,COEF_PROP)

         --==========
         mode='Caus'
         batch = getRandomBatchFromSeparateList(BATCH_SIZE,mode, USE_CONTINUOUS)
         LOSS_CAUS,gradCaus=doStuff_Caus(Models,caus_criterion,batch,COEF_CAUS)

         --==========
         mode='Rep'
         batch=getRandomBatchFromSeparateList(BATCH_SIZE,mode, USE_CONTINUOUS)
         LOSS_REP,gradRep=doStuff_Rep(Models,rep_criterion,batch,COEF_REP)
      end

      return LOSS_REP+LOSS_CAUS+LOSS_PROP+LOSS_TEMP ,gradParameters
    end

    --sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }
    --parameters, loss=optim.sgd(feval, parameters, sgdState)
    optimState={learningRate=LR, learningRateDecay=LR_DECAY}

    if SGD_METHOD == 'adagrad' then
        parameters,loss=optim.adagrad(feval,parameters,optimState)
    else
        parameters,loss=optim.adam(feval,parameters,optimState)
    end

    -- loss[1] table of one value transformed in just a value
    -- grad[1] we use just the first gradient to print the figure (there are 2 or 4 gradient normally)
    return loss[1], grad
end

function train_Epoch(Models,Prior_Used,LR, USE_CONTINUOUS)
    local NB_BATCHES= math.ceil(NB_SEQUENCES*AVG_FRAMES_PER_RECORD/BATCH_SIZE/(4+4+2+2))
    --90 is the FRAMES_PER_RECORD (average number of images per sequences for mobileRobot data), div by 12 because the network sees 12 images per iteration (i.e. record)
    -- (4*2 for rep and prop, 2*2 for temp and caus)

    local REP_criterion=get_Rep_criterion()
    local PROP_criterion=get_Prop_criterion()
    local CAUS_criterion=get_Caus_criterion()
    local TEMP_criterion=nn.MSDCriterion()

    local Temp_loss_list, Prop_loss_list, Rep_loss_list, Caus_loss_list = {},{},{},{}
    local Temp_loss_list_test,Prop_loss_list_test,Rep_loss_list_test,Caus_loss_list_test = {},{},{},{}
    local Sum_loss_train, Sum_loss_test = {},{}
    local Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list = {},{},{},{}
    local list_errors,list_MI, list_corr={},{},{}

    local Prop=Have_Todo(Prior_Used,'Prop') --TODOrename applies_prior()
    local Temp=Have_Todo(Prior_Used,'Temp')
    local Rep=Have_Todo(Prior_Used,'Rep')
    local Caus=Have_Todo(Prior_Used,'Caus')
    print(Prop)
    print(Temp)
    print(Rep)
    print(Caus)

    local coef_Temp=1
    local coef_Prop=1
    local coef_Rep=1
    local coef_Caus=1
    local coef_list={coef_Temp,coef_Prop,coef_Rep,coef_Caus}

    print(NB_SEQUENCES..' : sequences. '..NB_BATCHES..' batches')

    for epoch=1, NB_EPOCHS do
       print('--------------Epoch : '..epoch..' ---------------')
       local Temp_loss,Prop_loss,Rep_loss,Caus_loss=0,0,0,0
       local Grad_Temp,Grad_Prop,Grad_Rep,Grad_Caus=0,0,0,0

       xlua.progress(0, NB_BATCHES)
       for numBatch=1, NB_BATCHES do
          index1=torch.random(1,NB_SEQUENCES-1)
          index2=torch.random(1,NB_SEQUENCES-1)
          ------------- only one list used----------
          --       print([[====================================================
          -- WARNING TESTING PRIOR, THIS IS NOT RANDOM AT ALL
          -- ====================================================]])
          --       local index1=8
          --       local index2=3

          local data1 = load_seq_by_id(index1)
          local data2 = load_seq_by_id(index2)

          assert(data1, "Something went wrong while loading data1")
          assert(data2, "Something went wrong while loading data2")

          if Temp then
             Loss,Grad=Rico_Training(Models,'Temp',data1,data2,TEMP_criterion, coef_Temp,LR,BATCH_SIZE, USE_CONTINUOUS)
             Grad_Temp=Grad_Temp+Grad
             Temp_loss=Temp_loss+Loss
          end
          if Prop then
             Loss,Grad=Rico_Training(Models,'Prop',data1,data2, PROP_criterion, coef_Prop,LR,BATCH_SIZE, USE_CONTINUOUS)
             Grad_Prop=Grad_Prop+Grad
             Prop_loss=Prop_loss+Loss
          end
          if Rep then
             Loss,Grad=Rico_Training(Models,'Rep',data1,data2,REP_criterion, coef_Rep,LR,BATCH_SIZE, USE_CONTINUOUS)
             Grad_Rep=Grad_Rep+Grad
             Rep_loss=Rep_loss+Loss
          end
          if Caus then
             Loss,Grad=Rico_Training(Models,'Caus',data1,data2,CAUS_criterion,coef_Caus,LR,BATCH_SIZE, USE_CONTINUOUS)
             Grad_Caus=Grad_Caus+Grad
             Caus_loss=Caus_loss+Loss
          end
          xlua.progress(numBatch, NB_BATCHES)
       end

       local id=name..epoch -- variable used to not mix several log files

       print("Loss Temp", Temp_loss/NB_BATCHES/BATCH_SIZE)
       print("Loss Prop", Prop_loss/NB_BATCHES/BATCH_SIZE)
       print("Loss Caus", Caus_loss/NB_BATCHES/BATCH_SIZE)
       print("Loss Rep", Rep_loss/NB_BATCHES/BATCH_SIZE)
       print("Saving model in ".. NAME_SAVE)
       save_model(Models.Model1, NAME_SAVE)
   end
end

local records_paths = Get_Folders(DATA_FOLDER, 'record') --local list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)
NB_SEQUENCES= #records_paths
if NB_SEQUENCES ==0  then --or not folder_exists(DATA_FOLDER) then
    error('Error: data was not found in input directory INPUT_DIR= '.. DATA_FOLDER)
end

if CAN_HOLD_ALL_SEQ_IN_RAM then
   print("Preloading all sequences in memory, that way, to accelerate batch selection")
   ALL_SEQ = {} -- Preload all the sequences instead of loading specific sequences during batch selection
   for id=1,NB_SEQUENCES do
      ALL_SEQ[#ALL_SEQ+1] = load_seq_by_id(id)
   end
end

for nb_test=1, #PRIORS_TO_APPLY do
   if RELOAD_MODEL then
      print("Reloading model in "..SAVED_MODEL_PATH)  --TODO: undefined constant MODEL_FILE_STRING, use SAVED_MODEL_PATH = NAME_SAVE?
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

   local Priors= PRIORS_TO_APPLY[nb_test]
   local Log_Folder=Get_Folder_Name(LOG_FOLDER, Priors)
   print("Training epoch : "..nb_test ..' using Log_Folder: '..Log_Folder)
   train_Epoch(Models,Priors, LR, USE_CONTINUOUS)
end

imgs={} --memory is free!!!!!
