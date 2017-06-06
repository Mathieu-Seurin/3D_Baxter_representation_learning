--[[

  THIS IS A 3D VERSION OF THE 1D VERSION IN testRepresentations.lua
 BASELINE
]]--

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'xlua'   -- xlua provides useful tools, like progress bars
require 'nngraph'
require 'image'
require 'Get_Images_Set' -- for images_Paths(Path) Get_HeadCamera_View_Files
require 'functions'
require 'const'
require 'printing' --for show_figure
require 'definition_priors'
require 'optim_priors'
require 'lfs'

require 'math'
require 'string'
require 'MSDC'

-----------------SETTINGS
--nb_part = 50
-- if not USE_CUDA then
-- 	--	If there is RAM memory problems, one can try to split the dataset in more parts in order to load less image into RAM at one time.
-- 	--  by making "nb_part" larger than 50: -- ToDo: find a value less than 80 and more than 50 for data_baxter_short_seqs and <100 for data_baxter?
-- 	nb_part= 60
-- 	--MODEL_ARCHITECTURE_FILE ='./models/minimalNetModel' -- './models/topTripleFM_Split'
-- 	--BATCH_SIZE= 1
-- --else
-- 	--MODEL_ARCHITECTURE_FILE ='./models/topTripleFM_Split'
-- 	--BATCH_SIZE = 2 --60
-- end


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


---------------------------------------
--MODEL_NAME, name = 'Save97Win/reprLearner1d.t7', '97'
--MODEL_NAME,name = 'reprLearner1dWORKS.t7', 'works'
MODEL_NAME, representationsName = 'reprLearner3d.t7', 'default'  --TODO create
-- if this doesn't exist, it means you didn't run 'script.lua'
MODEL_FULL_PATH = MODEL_PATH..MODEL_NAME
DATA = PRELOAD_FOLDER..'imgsCv1.t7'  --TODO 'preloaded_simpleData3D_Seq1_normalized.t7'
PLOT = true

print('Running main script with USE_CUDA flag: '..tostring(USE_CUDA))
print('DIMENSION_OUT: '..DIMENSION_OUT.." LearningRate: "..LR.." BATCH_SIZE: "..BATCH_SIZE..". Using data folder: "..DATA_FOLDER.." Model file Torch: "..MODEL_ARCHITECTURE_FILE..'Preloaded DATA: '..DATA)

local function getReprFromImgs(imgs, PRELOAD_FOLDER, epresentations_name, model_full_path)
  -- we save all metrics that are going to be used in the network for
  -- efficiency (images that fulfill the criteria for each prior and their stats
  -- such as mean and std to avoid multiple computations )
   local fileName = PRELOAD_FOLDER..'allReprSaved'..representations_name..'.t7'

   if file_exists(fileName) then
      return torch.load(fileName)
   else
      print('Preloaded model does not exist: '..fileName..' Run train.lua first! ')
      os.exit()
   end
   print("Calculating all 3D representations with the model: "..fileName)
   print("Number of sequences to calculate :"..#imgs..' in BATCH_SIZE: '..BATCH_SIZE)

   X = {}
   print('getReprFromImgs by loading model: '..MODEL_PATH..MODEL_NAME)
   local model = torch.load(model_full_path)
   for numSeq,seq in ipairs(imgs) do
      print("numSeq",numSeq)
      for i,img in ipairs(seq) do
         x = nn.utils.addSingletonDimension(img)
         X[#X+1] = model:forward(x)[1]
      end
   end
   Xtemp = torch.Tensor(X)
   X = torch.zeros(#X,1)
   X[{{},1}] = Xtemp
   torch.save(fileName,X)
   return X
end

local function HeadPosFromTxts(txts, isData)
   --Since i use this function for creating X tensor for debugging
   -- or y tensor, the label tensor, i need a flag just to tell if i need X or y
   --isData = true => X tensor      isData = false => y tensor
   T = {}
   for l, txt in ipairs(txts) do
      truth = getTruth(txt)
      for i, head_pos in ipairs(truth) do
         T[#T+1] = head_pos
      end
   end
   T = torch.Tensor(T)

   if isData then --is it X or y that you need ?
      Ttemp = torch.zeros(T:size(1),1)
      Ttemp[{{},1}] = T
      T = Ttemp
   end
   return T
end
	  for l, txt in ipairs(txts) do
	     truth = getTruth(txt)
	     for i, head_pos in ipairs(truth) do
	        if head_pos < 0.1 and head_pos > -0.1 then
	           y[#y+1] = 1
	        else
	           y[#y+1] = 2
	        end
	     end
	  end
    return torch.Tensor(y)
end

local function RandomBatch(X,y,BATCH_SIZE)
   local numSeq = X:size(1)
   batch = torch.zeros(BATCH_SIZE,1)
   y_temp = torch.zeros(BATCH_SIZE)

   for i=1,BATCH_SIZE do
      local id=torch.random(1,numSeq)
      batch[{i,1}] = X[{id,1}]
      y_temp[i] = y[id]
   end
   -- print("batch",batch)
   -- print("y_temp",y_temp)
   -- io.read()
   if USE_CUDA then
     batch = batch:cuda()
     y_temp = y_temp:cuda()
   end
   return batch, y_temp
end

function Rico_Training_evaluation(model,batch,y,reconstruct, LR, USE_CONTINUOUS) --TODO USE ONLY ONE, SAME AS IN SCRIPT-> move to functions?
   local criterion
   local optimizer = optim.adam
   if reconstruct then
      if USE_CUDA then
        criterion = nn.SmoothL1Criterion():cuda()
      else
        criterion = nn.SmoothL1Criterion()
      end
   else
     if USE_CUDA then
      criterion = nn.CrossEntropyCriterion():cuda()
     else
      criterion = nn.CrossEntropyCriterion()
     end
   end

   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
      -- just in case:
      collectgarbage()
      -- get new parameters
      if x ~= parameters then
         parameters:copy(x)
      end
      -- reset gradients
      gradParameters:zero()
      local yhat = model:forward(batch)
      local loss = criterion:forward(yhat,y)
      local grad = criterion:backward(yhat,y)
      model:backward(batch, grad)

      return loss,gradParameters
   end
   optimState={learningRate=LR}
   parameters, loss=optimizer(feval, parameters, optimState)
   return loss[1]
end

function accuracy(X_test,y_test,model)
   local acc = 0
   if USE_CUDA then
    local yhat = model:forward(X_test:cuda())
   else
    local yhat = model:forward(X_test)
  end

   _,yId = torch.max(yhat,2)
   for i=1,X_test:size(1) do
      if yId[i][1]==y_test[i] then
         acc = acc + 1
      end
   end
   return acc/y_test:size(1)
end

function accuracy_reconstruction(X_test,y_test, model)
   local acc = 0
   if USE_CUDA then
    local yhat = model:forward(X_test:cuda())
   else
    local yhat = model:forward(X_test)
   end
   -- print("yhat",yhat[1][1],yhat[2][1],yhat[3][1],yhat[4][1],yhat[60][1])
   -- print("y",truth[1],truth[2],truth[3],truth[4],truth[60])

   for i=1,X_test:size(1) do
      acc = acc + math.sqrt(math.pow(yhat[i][1]-y_test[i],2))
   end
   return acc/X_test:size(1)
end

function rand_accuracy(y_test)
   count = 0
   for i=1,y_test:size(1) do
      if y_test[i]==2 then
         count = count + 1
      end
   end
   return count/y_test:size(1)
end

function createModelReward()
   net = nn.Sequential()
   net:add(nn.Linear(1,3))
   net:add(nn.Tanh())
   net:add(nn.Linear(3,2))
   if USE_CUDA then
    return net:cuda()
   else
    return net
   end
end

function createModelReconstruction()
   net = nn.Sequential()
   net:add(nn.Linear(1,1))
   if USE_CUDA then
    return net:cuda()
   else
    return net
   end
end

function train(X,y, reconstruct, NB_SEQUENCES)
   reconstruct = reconstruct or true

   --local nbList = 10  --TODO nbList= #list_folders_images?
   local numEx = X:size(1)
   local splitTrainTest = 0.75

   local sizeTest = math.floor(numEx/NB_SEQUENCES)

   id_test = {{math.floor(numEx*splitTrainTest), numEx}}
   X_test = X[id_test]
   y_test = y[id_test]

   id_train = {{1,math.floor(numEx*splitTrainTest)}}
   X_train = X[id_train]
   y_train = y[id_train]

   if reconstruct then
      model = createModelReconstruction()
      print("Test accuracy before training",accuracy_reconstruction(X_test,y_test,model))

   else
      model = createModelReward()
      print("Test accuracy before training",accuracy(X_test,y_test,model))
      print("Random accuracy", rand_accuracy(y_test))
   end
   parameters,gradParameters = model:getParameters()

   for epoch=1, NB_EPOCH do
      local lossTemp=0
      for numBatch=1, NB_BATCHES do
         batch_temp, y = RandomBatch(X_train,y_train,BATCH_SIZE)
         lossTemp = lossTemp + Rico_Training_evaluation(model,batch_temp,y, reconstruct, LR)
      end

      if epoch==NB_EPOCH then
         print("lossTemp",lossTemp/NB_BATCHES)

         if reconstruct then
            print("Test accuracy = ",accuracy_reconstruction(X_test,y_test,model))
         else
            print("Test accuracy = ",accuracy(X_test,y_test,model))
         end
      end
   end
end

function createPreloadedDataFolder(list_folders_images,list_txt,LOG_FOLDER,use_simulate_images,LR, model_full_path)
  --  local BATCH_SIZE=16 -- TODO use GLOBALs
  --  local NB_EPOCHS=2
   --local totalBatch=20
   --local name_save=LOG_FOLDER..'reprLearner1d.t7'
   local coef_Temp=1
   local coef_Prop=1
   local coef_Rep=1
   local coef_Caus=2
   local coef_list={coef_Temp,coef_Prop,coef_Rep,coef_Caus}
   local list_corr={}

  --  local plot = false    local loading = true

   NB_SEQUENCES = #list_folders_images
	 print('createPreloadedDataFolder for NB_SEQUENCES: '..NB_SEQUENCES)
   --local part = 1 --
   local next_part_start_index = 1
   for crossValStep=1, NB_SEQUENCES do
      models = createModels(model_full_path)
      currentLogFolder=LOG_FOLDER..'CrossVal'..crossValStep..'/' --*
      current_preload_file = PRELOAD_FOLDER..'imgsCv'..crossValStep..'.t7'

      if file_exists(current_preload_file) and RELOAD_MODEL then
         print("Preloaded Data Already Exists, Loading from file: "..current_preload_file.."...")
         imgs = load_seq_by_id(crossValStep)--imgs = torch.load(current_preload_file)
         local imgs_test = imgs[#imgs]
				 print('imgs and imgs_test')
				 print (#imgs)
				 print(#imgs_test)
      else
         print("Preloaded Data Does Not Exists. Loading Training and Test from "..DATA.." and saving to "..current_preload_file)

		 local imgs = torch.load(DATA) --local imgs, imgs_test = loadTrainTest(list_folders_images,crossValStep, PRELOAD_FOLDER)
		--  if crossValStep ==NB_SEQUENCES then
		-- 	 test_sequence_index = crossValStep +1
		--  else
		-- 	 test_sequence_index = 1
		--  end
		 local imgs_test = load_seq_by_id(crossValStep)
		 print ("imgs "..#imgs)
		 print ("load_seq_by_id #imgs_test: "..#imgs_test)

		 imgs[1], imgs[#imgs] = imgs[#imgs], imgs[1] -- Because during database creation we swapped those values

		 torch.save(current_preload_file, imgs)
      end

      -- we use last list as test
      list_txt[crossValStep],list_txt[#list_txt] = list_txt[#list_txt], list_txt[crossValStep]
      local txt_test=list_txt[#list_txt]
      local truth, next_part_start_index = get_true_hand_position_3D(txt_test)--,nb_part, next_part_start_index)-- getTruth(txt_test,use_simulate_images) --for the 1D case
      -- print (txt_test)
      -- print(list_txt)

      assert(#imgs_test==#truth,"Different number of images and corresponding ground truth, something is wrong \nNumber of Images : "..#imgs_test.." and Number of truth values : "..#truth)

      if plot then
         show_figure(truth,currentLogFolder..'GroundTruth.log')
      end
      -- corr=Print_performance(models, imgs_test,txt_test,"First_Test",currentLogFolder,truth,false)
      -- print("Correlation before training : ", corr)
      -- table.insert(list_corr,corr)

			NB_BATCHES = math.floor(#imgs/BATCH_SIZE)
			print("Training with NB_SEQUENCES "..NB_SEQUENCES..' NB_BATCHES: '..NB_BATCHES)
      for epoch=1, NB_EPOCHS do
         print('--------------Epoch : '..epoch..' ---------------')
         local lossTemp=0
         local lossRep=0
         local lossProp=0
         local lossCaus=0
         local causAdded = 0

         for numBatch=1,NB_BATCHES do
            indice1= torch.random(1,NB_SEQUENCES)
            repeat indice2= torch.random(1,NB_SEQUENCES) until (indice1 ~= indice2)

            txt1=list_txt[indice1]
            txt2=list_txt[indice2]

            imgs1=imgs[indice1]
            imgs2=imgs[indice2]
						-- local data1 = load_seq_by_id(indice1)
						-- local data2 = load_seq_by_id(indice2)

            batch=getRandomBatchFromSeparateList(BATCH_SIZE,'Temp', USE_CONTINUOUS)--batch(imgs1,imgs2,txt1,txt2,BATCH_SIZE,"Temp")
            lossTemp = lossTemp + Rico_Training_evaluation(models,'Temp',batch, coef_Temp,LR)

            batch=getRandomBatchFromSeparateList(BATCH_SIZE,'Caus', USE_CONTINUOUS)
            lossCaus = lossCaus + Rico_Training_evaluation(models, 'Caus',batch, 1,LR)

            batch=getRandomBatchFromSeparateList(BATCH_SIZE,'Prop', USE_CONTINUOUS)
            lossProp = lossProp + Rico_Training_evaluation(models, 'Prop',batch, coef_Prop,LR)

            batch=getRandomBatchFromSeparateList(BATCH_SIZE,'Rep', USE_CONTINUOUS)
            lossRep = lossRep + Rico_Training_evaluation(models,'Rep',batch, coef_Rep,LR)

            xlua.progress(numBatch, NB_BATCHES)

         end
         --corr=Print_performance(models, imgs_test,txt_test,"Test",currentLogFolder,truth,false)
         print("lossTemp",lossTemp/totalBatch)
         print("lossProp",lossProp/totalBatch)
         print("lossRep",lossRep/totalBatch)
         print("lossCaus",lossCaus/(totalBatch+causAdded))
      end
      corr=Print_performance(models, imgs_test,txt_test,"Test",currentLogFolder,truth,plot)
      --show_figure(list_corr,currentLogFolder..'correlation.log','-')

      --for reiforcement, we need mean and std to normalize representation
      print("SAVING MODEL AND REPRESENTATIONS")
      saveMeanAndStdRepr(imgs)
      models.model1:float()
      --save_model(models.model1, name_save) --TODO
      list_txt[crossValStep],list_txt[#list_txt] = list_txt[#list_txt], list_txt[crossValStep]
   end
end

--from functions 1D
function createModels(MODEL_FULL_PATH)
   if RELOAD_MODEL then
      print("Loading Model..."..MODEL_FULL_PATH)
      if file_exists(MODEL_FULL_PATH) then
         model = torch.load(MODEL_FULL_PATH)  --LOG_FOLDER..'20e.t7'
      else
         print("Model file does not exist!")
         os.exit()
      end
   else
      model=getModel(DIMENSION_OUT)
      print(model)
   end

   if USE_CUDA then
		 model=model:cuda()
	 else
		 model=model:double()
	 end
   parameters,gradParameters = model:getParameters()
   model2=model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   model3=model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   model4=model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   models={model1=model,model2=model2,model3=model3,model4=model4}
   return models
end

-- function getTruth(txt)
--    local truth={}
--    local head_pan_indice=2
--    local tensor, label=tensorFromTxt(txt)
--
--    for i=1, (#tensor[{}])[1] do
--       table.insert(truth, tensor[i][head_pan_indice])
--    end
--    return truth
-- end
---------------------------------------------------------------------------------------
-- Function : getTruth(txt,use_simulate_images)   3D function
-- Input (txt) :
-- Input (use_simulate_images) :
-- Input (arrondit) :
-- Output (truth):
---------------------------------------------------------------------------------------
function get_true_hand_position_3D(data_file_with_hand_pos)
	--DO NOT USE SHOULD BE DONE ALREADY IN getInfos
	print ('get_true_hand_position_3D tensor and label: ')
	local truth={}
	local tensor, label=tensorFromTxt(data_file_with_hand_pos)
	--print (tensor) -- tensor and truth are a DoubleTensor of size 100*4
	print ((#tensor[{}])[1])
	for i=1, (#tensor[{}])[1] do
	   table.insert(truth, tensor[i]) --[2,3,4]?
	end
	--print("truth")
	--print (truth)
	return truth
end
-- function get_Truth_3D(txt_joint, nb_part, part)
-- 	local x=2
-- 	local y=3
-- 	local z=4
-- 	print ('get_Truth_3D for nb_part: '..nb_part..' txt_joint'..txt_joint)
-- 	part = 1
-- 	local tensor, label=tensorFromTxt(txt_joint)
-- 	local list_lenght = torch.floor((#tensor[{}])[1]/nb_part)
-- 	local start=list_lenght*part +1
--   local part_last_index = start+list_lenght
-- 	local list_truth={}
-- 	for i=start,part_last_index do--(#tensor[{}])[1] do
-- 		local truth=torch.Tensor(3)
-- 		truth[1]=tensor[i][x]
-- 		truth[2]=tensor[i][y]
-- 		truth[3]=tensor[i][z]
-- 		table.insert(list_truth,truth)
-- 	end
-- 	return list_truth, part_last_index
-- end

local function getHandPosFromTxts(txts, nb_part, part)
   --Since i use this function for creating X tensor for debugging
   -- or y tensor, the label tensor, i need a flag just to tell if i need X or y
   --isData = true => X tensor      isData = false => y tensor
   T = {}
   for l, txt in ipairs(txts) do
      truth, part = get_Truth_3D(txt, nb_part, part)
      for i, hand_pos in ipairs(truth) do
         T[#T+1] = hand_pos
      end
   end
   T = torch.Tensor(T)
   if isData then --is it X or y that you need ?
      Ttemp = torch.zeros(T:size(1),1)
      Ttemp[{{},1}] = T
      T = Ttemp
   end

   return T
end

local function getRewardsFromTxts(txt_joint, nb_parts, part)
   y = {}
    for l, txt in ipairs(txt_joint) do
       truth, part = get_Truth_3D(txt_joint, nb_part, part)
       for i, head_pos in ipairs(truth) do
          if head_pos.x < 0.1 and head_pos.x > -0.1 then
						--TODO: get real positions of the button
             y[#y+1] = 1  -- negative vs positive reward
          else
             y[#y+1] = 2
          end
       end
    end
   return torch.Tensor(y)
end




------------------------------------
-- Our representation learnt should be coordinate independent, as it is not aware of
-- what is x,y,z and thus, we should be able to reconstruct the state by switching
-- to y,x,z, or if we add Gaussian noise to the true positions x,y,z of Baxter arm.
-- These will be our score baseline to compare with
reconstructingTask = true
local dataAugmentation=true
--local list_folders_images, list_txt=Get_HeadCamera_HeadMvt() local _, list_txt=Get_HeadCamera_HeadMvt(DATA_FOLDER)

-- if not file_exists(PRELOAD_FOLDER) then   --TODO  needed? Remove and leave in const only once
--    lfs.mkdir(PRELOAD_FOLDER)
-- end
-- if not file_exists(LOG_FOLDER) then
--    lfs.mkdir(LOG_FOLDER)
-- end
---
list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)
NB_SEQUENCES = #list_folders_images
print(NB_SEQUENCES..' data sequences')
indice_test = 1--
-- print("Got list_folders_images: ")
-- print(list_folders_images)
avg_error = 0
seq_id = 1 --TODO provide for each sequence and give final avg reconstruction error  local indice_test = torch.random(1,NB_SEQUENCES-1)

if #list_folders_images >0 then
	local images_in_path= images_Paths(list_folders_images[indice_test])
	print('images_paths (First test:)'..list_folders_images[indice_test])
	txt_test=list_txt_state[indice_test]
	txt_reward_test=list_txt_button[indice_test]
	--part_test=1
	--Data_test=load_Part_list(images_in_path,txt_test,txt_reward_test,image_width,image_height,nb_part,part_test,0,txt_test) --avoid, call only  load_seq_by_id(seq_id)
	--Data_test=load_Part_list(list,txt,txt_reward,IM_LENGTH,IM_HEIGHT,DATA_AUGMENTATION,txt_state)
	data_test=load_seq_by_id(seq_id)
	print ('data_test: '..#data_test)
	assert(data_test,'Something went wrong while loading data1')
	local truth = get_true_hand_position_3D(txt_test)
	 --get_Truth_3D(txt_test,nb_part,part_test) -- 100 DoubleTensor of size 3
	print (truth)
	print("Plotting the truth... ")
	--show_figure(truth, LOG_FOLDER..'The_Truth.Log','Truth',data_test.Infos)
	print("Computing performance... ")
	--Print_performance(Models, data_test,txt_test,txt_reward_test,"First_Test",LOG_FOLDER,truth)
	---
	print(list_txt_state)

	createPreloadedDataFolder(list_folders_images,list_txt_state,LOG_FOLDER,use_simulate_images,LR,MODEL_FULL_PATH)
	imgs={}
	------------------------------------
	print('load data from DATA ',DATA)
	local imgs = torch.load(DATA)
	imgs[1], imgs[#imgs] = imgs[#imgs], imgs[1] -- Because during database creation we swapped those values

	if reconstructingTask then
	   y = getHandPosFromTxts(list_txt, NB_BATCHES, 1)-- nb_part, 1)
	else
	   y = getHandPosRewardsFromTxts(list_txt, NB_BATCHES, 1)--nb_part, 1)
	end

	--X = HeadPosFromTxts(list_txt,true)
	-- predict:
	X = getReprFromImgs(imgs, PATH_PRELOAD_DATA,representationsName, MODEL_FULL_PATH)

	train(X,y, reconstructingTask, NB_SEQUENCES)
else
	print("Input image files not found, check your DATA_FOLDER global variable (should be named 'simpleData3D')")
	os.exit()
end
