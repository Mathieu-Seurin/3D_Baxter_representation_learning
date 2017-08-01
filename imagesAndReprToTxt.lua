require 'lfs'
require 'torch'
require 'image'
require 'nn'
require 'nngraph'
require 'const'
require 'functions'
tnt = require 'torchnet'
vision = require 'torchnet-vision'

---------------------------------------------------------------------------------------
--This function uses the Torch module vision which preprocesses images. When
--calling functions in this way inside another method, we are using a pipeline
--(see torch pipelines documentation). This method will apply the transform composed
--necessary step as preprocessing for the image to be able to be fed into the network
--architecture model. ColorNormalize requires mean and std and so we set it. We finally
--transform the image from double to float() as it is required for our model in models/ folder.
---------------------------------------------------------------------------------------
local augmentation = tnt.transform.compose{
   vision.image.transformimage.colorNormalize{
      mean = MEAN_MODEL, std  = STD_MODEL
   },
   function(img) return img:float() end
}

function visualize_images_and_repr(imagesFolder,model)

   save_folder = imagesFolder..'images_with_repr/'
   lfs.mkdir(save_folder)

   for dir_seq_str in lfs.dir(imagesFolder) do
      if string.find(dir_seq_str,'record') then
         local imagesPath = imagesFolder..'/'..dir_seq_str..'/'
         print("ImagesPath",imagesPath)
         for imageStr in lfs.dir(imagesPath) do
            if string.find(imageStr,'png') then
               local fullImagesPath = imagesPath..imageStr

               if DIFFERENT_FORMAT then
                  img = image.scale(image.load(fullImagesPath,3,'float'), IM_LENGTH, IM_HEIGHT)
                  img = augmentation(img)
                  --print('DIFFERENT_FORMAT.. Doing augmentation...')
               else
                  img = getImageFormated(fullImagesPath)
                  --print('DIFFERENT_FORMAT is false: Formatting image only..') TODO Warn and check below that the autoencoder model minimalNetModel is not used together with priors training
               end

               img = img:double():reshape(1,IM_CHANNEL,IM_LENGTH,IM_HEIGHT)

               if USE_CUDA then
                  img = img:cuda()
               end
               --print ('img dimensions and model')     print (#img)                print(model)

               repr = model:forward(img)
               reprStr = ''
               for i=1,repr:size(2) do
                  reprStr = reprStr..arrondit(repr[{1,i}],0.01)..' '
               end

               img = image.scale(image.load(fullImagesPath,3,'float'), IM_LENGTH, IM_HEIGHT)

               -- print("",img:size())
               -- image.display(img[1])
               -- io.read()
               img = image.drawText(img,reprStr, 1, 1,{color = {255,255,255}})
               image.save(save_folder..imageStr,img)
            end


         end
      end
   end
end

--returns the representation of the image (a tensor of size {1xDIM})
function represent_all_images(imagesFolder, model)
   tempSeq = {}
   for dir_seq_str in lfs.dir(imagesFolder) do
      if string.find(dir_seq_str,'record') then
         print("Data sequence folder: ",dir_seq_str)
         local imagesPath = imagesFolder..'/'..dir_seq_str..'/'..SUB_DIR_IMAGE
         for imageStr in lfs.dir(imagesPath) do
            if string.find(imageStr,'jpg') then
               local fullImagesPath = imagesPath..'/'..imageStr
               local reprStr = ''

               if DIFFERENT_FORMAT then
                  img = image.scale(image.load(fullImagesPath,3,'float'), IM_LENGTH, IM_HEIGHT)
                  img = augmentation(img)
                  --print('DIFFERENT_FORMAT.. Doing augmentation...')
               else
                  img = getImageFormated(fullImagesPath)
                  --print('DIFFERENT_FORMAT is false: Formatting image only..') TODO Warn and check below that the autoencoder model minimalNetModel is not used together with priors training
               end

               img = img:double():reshape(1,IM_CHANNEL,IM_LENGTH,IM_HEIGHT)

               if USE_CUDA then
                  img = img:cuda()
               end
               --print ('img dimensions and model')     print (#img)                print(model)

               repr = model:forward(img)
               --    print ('repr')
               --    print(repr)
               for i=1,repr:size(2) do
                  reprStr = reprStr..repr[{1,i}]..' '
                  --print (reprStr)
               end
               tempSeq[#tempSeq+1] = {fullImagesPath, fullImagesPath..' '..reprStr}
               -- we write to file only second part of the tuple and use the first as key to sort them
            end
         end
      end
   end

   return tempSeq
end  --TODO call predict and add predict to script?


local function main(params)
   print("\n\n>> imagesAndReprToTxt.lua")
   set_hyperparams(params, '', false) --    print('In DATA_FOLDER: '..DATA_FOLDER..' params: ')- Overridden by the model loaded, therefore not used here: print(params)
   print_hyperparameters(false, 'imagesAndReprToTxt.lua Hyperparams')

   local images_folder = DATA_FOLDER
   local path, modelString
   folder_and_name = get_last_used_model_folder_and_name()
   path = folder_and_name[1]
   modelString = folder_and_name[2]
   savedModel = path..'/'..modelString  --t7 file
   print('Last model used: '..savedModel)

   -- if get_last_architecture_used(modelString) == 'AE' then
   --    assert(not(DIFFERENT_FORMAT), "For training the auto-encoder, the model architecture needs to be in the same format as BASE_TIMNET. Change in hyperparams.lua")
   --    print 'Overriding MODEL_ARCHITECTURE_FILE with BASE_TIMNET (only valid model for AE)'
   --    MODEL_ARCHITECTURE_FILE = BASE_TIMNET
   -- end

   -- NOT USEFUL ANYMORE : AE uses resnet now
    if not file_exists(savedModel) then
       print('SAVE_MODEL_T7_FILE = needs to be true (NECESSARY STEP TO RUN FULL EVALUATION PIPELINE (REQUIRED FILE BY imagesAndReprToTxt.lua)')
       error(savedModel.." file should exist")
    else
       local  model = torch.load(savedModel)
       if USE_CUDA then
          model = model:cuda()
       else
          model = model:double()
       end

       if params.visualize_seq then
          visualize_images_and_repr('saved_seq', model)
       else
          outStr = ''
          tempSeq = {}
          tempSeq = represent_all_images(images_folder, model)

          table.sort(tempSeq, function (a,b) return a[1] < b[1] end)
          tempSeqStr = ''
          for key in pairs(tempSeq) do
             tempSeqStr = tempSeqStr..tempSeq[key][2]..'\n'
          end
          path_to_output_file = path..'/'..LEARNED_REPRESENTATIONS_FILE

          print('Saving images and their learnt representations to file '..path_to_output_file)
          file = io.open(path_to_output_file, 'w')
          file:write(tempSeqStr)
          file:close()
       end
    end
end

-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-use_cuda', false, 'true to use GPU, false (default) for CPU only mode')
cmd:option('-use_continuous', false, 'true to use a continuous action space, false (default) for discrete one (0.5 range actions)')
cmd:option('-data_folder', MOBILE_ROBOT, 'Possible Datasets to use: staticButtonSimplest, mobileRobot, staticButtonSimplest, simpleData3D, pushingButton3DAugmented, babbling')
cmd:option('-mcd', 0.4, 'Max. cosine distance allowed among actions for priors loss function evaluation (MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)')
cmd:option('-sigma', 0.4, "Sigma: denominator in continuous actions' extra factor (CONTINUOUS_ACTION_SIGMA)")
cmd:option('-visualize_seq', false, "instead of computing representation for all images, just visualize a few for debugging purpose")

local params = cmd:parse(arg)
main(params)
