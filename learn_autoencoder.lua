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

function AE_Training(model, batch, optimizer)
   input=batch
   expected=batch

   if NOISE then
      noise=torch.rand(batch:size())
      noise = noise/3
      noise=noise:cuda() --this step is always needed when running in either CPU/GPU mode
      input=input+noise
   end

   -- local img_merge = image.toDisplayTensor({input[1],expected[1]})
   -- image.display{image=img_merge,win=WINDOW}
   -- io.read()

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

function train_Epoch(optimizer, list_folders_images,list_txt,Log_Folder)
   local totImg=AVG_FRAMES_PER_RECORD*NB_SEQUENCES

   print("total Imges: ",totImg)
   local nbIter=math.floor(totImg/BATCH_SIZE)

   local list_loss={}
   local list_corr={}
   local loss=0

   print("Begin Learning...")
   for epoch=1,NB_EPOCHS do
      loss=0
      print('--------------Epoch : '..epoch..' ---------------')
      for iter=1, nbIter do
         batch=getRandomBatchFromSeparateList(BATCH_SIZE, 'regular') --just taking random images from all sequences
         batch=batch:cuda()

         loss_iter=AE_Training(model,batch, optimizer)
         loss = loss + loss_iter
         xlua.progress(iter, nbIter)
      end
      print("DAE Mean Loss : "..loss/(nbIter*BATCH_SIZE))
      save_autoencoder(model,name_save)

      if epoch > 15 then
         test_model(model, list_folders_images)
      end
   end
end

function set_AE_hyperparams(params)
    -- OVERRIDING hyperparameters since it's not for auto-encoders :  ** MAIN DIFFERENCES:
    MODEL_ARCHITECTURE_FILE = BASE_TIMNET -- important to call in this order, as DIFFERENT_FORMAT is defined based on this setting. TODO idea: Pass MODEL_ARCHITECTURE_FILE as default cmd param in which is different in each script?
    set_hyperparams(params)
    LR = 0.0001
    BATCH_SIZE=20
    NUM_HIDDEN = 20
    NOISE = true
    if params.optimiser=="sgd" then  optimizer = optim.sgd end
    if params.optimiser=="rmsprop" then  optimizer = optim.rmsprop end
    if params.optimiser=="adam" then optimizer = optim.adam end
    return optimizer
end

local function main(params)
    print("\n\n>> learn_autoencoder.lua")
    optimizer = set_AE_hyperparams(params)
    print_hyperparameters()
    print(params)

    if DIFFERENT_FORMAT then
       error([[Don't forget to switch model to BASE_TIMNET in hyperparameters
    Because the images' format is the same for auto-encoder]])
    end

    NAME_SAVE = 'AE_'..NUM_HIDDEN..DATA_FOLDER
    print('NAME SAVE IN set_AE_hyp:  '..NAME_SAVE)
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
    train_Epoch(optimizer, list_folders_images,list_txt,Log_Folder)

    imgs={} --memory is free!!!!!
end


-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-optimiser', 'adam', 'Optimiser : adam|sgd|rmsprop')
cmd:option('-model', 'DAE', 'model : AE|DAE')
cmd:option('-use_cuda', false, 'true to use GPU, false (default) for CPU only mode')
cmd:option('-use_continuous', false, 'true to use a continuous action space, false (default) for discrete one (0.5 range actions)')
cmd:option('-data_folder', STATIC_BUTTON_SIMPLEST, 'Possible Datasets to use: staticButtonSimplest, mobileRobot, staticButtonSimplest, simpleData3D, pushingButton3DAugmented, babbling')

local params = cmd:parse(arg)
main(params)
