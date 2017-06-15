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
LR = 0.001
BATCH_SIZE=20
NUM_HIDDEN = DIMENSION_OUT
NORMALIZE_IMAGE = false
IM_LENGTH = 200
IM_HEIGHT = 200

if DIFFERENT_FORMAT then
   error([[Don't forget to switch model to BASE_TIMNET in hyperparameters
Because the images' format is the same for auto-encoder"]])
end


NAME_SAVE = 'AE_'..NUM_HIDDEN..DATA_FOLDER

function AE_Training(model,batch)
   if opt.optimiser=="sgd" then  optimizer = optim.sgd end
   if opt.optimiser=="rmsprop" then  optimizer = optim.rmsprop end
   if opt.optimiser=="adam" then optimizer = optim.adam end
   model:cuda()

   input=batch
   expected=batch

   if opt.model=="DAE" then
      noise=torch.rand(batch:size())
      noise=(noise-noise:mean())/(noise:std())
      noise=noise:cuda()
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
      criterion=nn.AbsCriterion()
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


function train_Epoch(list_folders_images,list_txt,Log_Folder)

   nbList= #list_folders_images
   local nbEpoch=5
   local totImg=AVG_FRAMES_PER_RECORD*nbList

   print("totImg",totImg)
   local nbIter=math.floor(totImg/BATCH_SIZE)

   local list_loss={}
   local list_corr={}
   local loss=0

   print("Begin Learning")
   for epoch=1, nbEpoch do
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
   end
end

-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-optimiser', 'adam', 'Optimiser : adam|sgd|rmsprop')
cmd:option('-model', 'DAE', 'model : AE|DAE')
opt = cmd:parse(arg)


torch.manualSeed(1337)
LR=0.001

local Log_Folder='./Log/'
local list_folders_images, list_txt=Get_HeadCamera_View_Files(DATA_FOLDER)

NB_SEQUENCES = #list_folders_images
ALL_SEQ = precompute_all_seq()

image_width=IM_LENGTH
image_height=IM_HEIGHT

require('./models/autoencoder_conv.lua')
model = getModel()
model=model:cuda()

parameters,gradParameters = model:getParameters()
train_Epoch(list_folders_images,list_txt,Log_Folder)

imgs={} --memory is free!!!!!
