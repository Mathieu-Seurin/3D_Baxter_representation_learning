require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'nngraph'
--local cuda = pcall(require, 'cutorch') -- Use CUDA if available

require 'Get_Images_Set'
require 'functions'

require 'const'

print("============ DATA USED =========\n",
                    DATA_FOLDER,
      "\n================================")

-- OVERRIDING hyperparameters since it's not for auto-encoders :
LR = 0.0001
BATCH_SIZE=20
NUM_HIDDEN = 20
NOISE = true

if DIFFERENT_FORMAT then
   error([[Don't forget to switch model to BASE_TIMNET in hyperparameters
Because the images' format is the same for auto-encoder]])
end


NAME_SAVE = 'AE_'..NUM_HIDDEN..DATA_FOLDER

function AE_Training(model,batch)
   if opt.optimiser=="sgd" then  optimizer = optim.sgd end
   if opt.optimiser=="rmsprop" then  optimizer = optim.rmsprop end
   if opt.optimiser=="adam" then optimizer = optim.adam end

   input=batch
   expected=batch

   if NOISE then
      noise=torch.rand(batch:size())
      noise=(noise-noise:mean())/(noise:std())
      if USE_CUDA then
         noise=noise:cuda()
    --   else
    --      noise=noise:double()
      end
      input=input+noise
   end

   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
      -- just in case:
      collectgarbage()
      --get new parameters
      if x ~= parameters then
         parameters:copy(x)
      end
      --reset gradients
      gradParameters:zero()
      criterion=nn.SmoothL1Criterion()
      criterion=criterion:cuda()
      ouput=model:forward(input)
      loss = criterion:forward(ouput, expected)
      gradInput=model:backward(input, criterion:backward(ouput, expected))
      --print(gradInput:mean())
      return loss,gradParameters
   end
   optimState={learningRate=LR}
   parameters, loss=optimizer(feval, parameters, optimState)

   return loss[1]
end

function test_model(model, list_folders_images)

   criterion=nn.SmoothL1Criterion()
   criterion:cuda()

   for num_test=0,NB_TEST-1 do
      local id_test = #list_folders_images - num_test
      local seq = load_seq_by_id(id_test)
      for im=1,#seq.images do
         current_img = seq.images[im]:view(1,3,200,200):cuda()
         output = model:forward(current_img)

         if VISUALIZE_AE and im == 1 then
            local img_merge = image.toDisplayTensor({current_img[1],output})
            image.display{image=img_merge,win=WINDOW}
            io.read()
         end
      end
   end
end

function train_Epoch(list_folders_images,list_txt,Log_Folder)

   local totImg=AVG_FRAMES_PER_RECORD*NB_SEQUENCES

   print("totImg",totImg)
   local nbIter=math.floor(totImg/BATCH_SIZE)

   local list_loss={}
   local list_corr={}
   local loss=0

   print("Begin Learning")
   for epoch=1,NB_EPOCHS do
      loss=0
      print('--------------Epoch : '..epoch..' ---------------')
      for iter=1, nbIter do
         batch=getRandomBatchFromSeparateList(BATCH_SIZE, 'regular') --just taking random images from all sequences
         batch=batch:cuda()

         loss_iter=AE_Training(model,batch)
         loss = loss + loss_iter
         xlua.progress(iter, nbIter)
      end
      print("Mean Loss : "..loss/(nbIter*BATCH_SIZE))
      save_autoencoder(model,name_save)

      if epoch > 15 then
         test_model(model, list_folders_images)
      end
   end
end

-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-optimiser', 'adam', 'Optimiser : adam|sgd|rmsprop')
cmd:option('-model', 'DAE', 'model : AE|DAE')
opt = cmd:parse(arg)

local list_folders_images, list_txt=Get_HeadCamera_View_Files(DATA_FOLDER)

NB_TEST = 3
NB_SEQUENCES = #list_folders_images-NB_TEST --That way, the last X sequences are used as test

ALL_SEQ = precompute_all_seq()

image_width=IM_LENGTH
image_height=IM_HEIGHT

require('./models/autoencoder_conv')
model = getModel()
model=model:cuda()

parameters,gradParameters = model:getParameters()
train_Epoch(list_folders_images,list_txt,Log_Folder)

imgs={} --memory is free!!!!!
