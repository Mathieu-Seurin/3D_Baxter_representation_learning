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
      local batch=getRandomBatchFromSeparateList(BATCH_SIZE,mode)
      LOSS_TEMP,grad=doStuff_temp(Models,temp_criterion, batch,COEF_TEMP)

      if USE_CONTINUOUS then
         --==========
         mode='Prop'
         batch, action1, action2 = getRandomBatchFromSeparateListContinuous(BATCH_SIZE,mode)
         LOSS_PROP,gradProp=doStuff_Prop_continuous(Models,prop_criterion,batch,COEF_PROP, action1, action2)

         --==========
         mode='Caus'
         batch, action1, action2 = getRandomBatchFromSeparateListContinuous(BATCH_SIZE,mode)
         LOSS_CAUS,gradCaus=doStuff_Caus_continuous(Models,caus_criterion,batch,COEF_CAUS, action1, action2)

         --==========
         mode='Rep'
         batch, action1, action2 = getRandomBatchFromSeparateListContinuous(BATCH_SIZE,mode)
         LOSS_REP,gradRep=doStuff_Rep_continuous(Models,rep_criterion,batch,COEF_REP, action1, action2)
      else
         --==========
         mode='Prop'
         batch = getRandomBatchFromSeparateList(BATCH_SIZE,mode)
         LOSS_PROP,gradProp=doStuff_Prop(Models,prop_criterion,batch,COEF_PROP)

         --==========
         mode='Caus'
         batch = getRandomBatchFromSeparateList(BATCH_SIZE,mode)
         LOSS_CAUS,gradCaus=doStuff_Caus(Models,caus_criterion,batch,COEF_CAUS)

         --==========
         mode='Rep'
         batch=getRandomBatchFromSeparateList(BATCH_SIZE,mode)
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

function train_Epoch(Models,Prior_Used,LOG_FOLDER,LR, USE_CONTINUOUS)
   local NB_BATCHES= math.ceil(NB_SEQUENCES*90/BATCH_SIZE/(4+4+2+2))
   --90 is the average number of images per sequences, div by 12 because the network sees 12 images per iteration
   -- (4*2 for rep and prop, 2*2 for temp and caus)
   
   print(NB_SEQUENCES..' : sequences. '..NB_BATCHES..' batches')

   for epoch=1, NB_EPOCHS do
      print('--------------Epoch : '..epoch..' ---------------')
      local total_temp_loss,total_prop_loss,total_rep_loss,total_caus_loss=0,0,0,0

      xlua.progress(0, NB_BATCHES)

      for numBatch=1, NB_BATCHES do

         Loss,Grad=Rico_Training(Models)

         total_temp_loss = total_temp_loss + LOSS_TEMP --Ugly, this variable is located in Rico_Training in eval
         total_caus_loss = total_caus_loss + LOSS_CAUS --Ugly, this variable is located in Rico_Training in eval
         total_rep_loss = total_rep_loss + LOSS_REP --Ugly, this variable is located in Rico_Training in eval
         total_prop_loss = total_prop_loss + LOSS_PROP --Ugly, this variable is located in Rico_Training in eval
         
         xlua.progress(numBatch, NB_BATCHES)
      end

      local id=name..epoch -- variable used to not mix several log files

      print("Loss Temp", total_temp_loss/NB_BATCHES/BATCH_SIZE)
      print("Loss Prop", total_prop_loss/NB_BATCHES/BATCH_SIZE)
      print("Loss Caus", total_caus_loss/NB_BATCHES/BATCH_SIZE)
      print("Loss Rep", total_rep_loss/NB_BATCHES/BATCH_SIZE)

      if USE_CONTINUOUS then
         model_name = NAME_SAVE..'Continuous'
      else
         model_name = NAME_SAVE
      end
      save_model(Models.Model1, model_name)
   end
end

Tests_Todo={
   {"Prop","Temp","Caus","Rep"}
   --[[
      {"Rep","Caus","Prop"},
      {"Rep","Caus","Temp"},
      {"Rep","Prop","Temp"},
      {"Prop","Caus","Temp"},
      {"Rep","Caus"},
      {"Prop","Caus"},
      {"Temp","Caus"},
      {"Temp","Prop"},
      {"Rep","Prop"},
      {"Rep","Temp"},
      {"Rep"},
      {"Temp"},
      {"Caus"},
      {"Prop"}
   --]]
}

local list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)
NB_SEQUENCES= #list_folders_images

if LOGGING_ACTIONS then
   print("LOGGING ACTIONS")
   LOG_ACTION = {}

   for i=1,NB_SEQUENCES do
      LOG_ACTION[#LOG_ACTION+1] = {}
   end
   
end

if CAN_HOLD_ALL_SEQ_IN_RAM then
   print("Preloading all sequences in memory, that way, to accelerate batch selection")
   ALL_SEQ = {} -- Preload all the sequences instead of loading specific sequences during batch selection
   for id=1,NB_SEQUENCES do
      ALL_SEQ[#ALL_SEQ+1] = load_seq_by_id(id)
   end
end


for nb_test=1, #Tests_Todo do

   if RELOAD_MODEL then
      Model = torch.load(MODEL_FILE_STRING):double()
   else
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

   local Priors=Tests_Todo[nb_test]
   local Log_Folder=Get_Folder_Name(LOG_FOLDER,Priors)
   print("Current test : "..LOG_FOLDER)
   train_Epoch(Models,Priors,Log_Folder,LR, USE_CONTINUOUS)
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
