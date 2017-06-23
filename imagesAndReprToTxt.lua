require 'lfs'
require 'torch'
require 'image'
require 'nn'
require 'nngraph'
require 'const'
require 'functions'

--returns the representation of the image (a tensor of size {1xDIM})
function represent_all_images(imagesFolder, model)
   local augmentation = tnt.transform.compose{
      vision.image.transformimage.colorNormalize{
         mean = MEAN_MODEL, std  = STD_MODEL
      },
      function(img) return img:float() end
   }

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
    set_hyperparams(params) --    print('In DATA_FOLDER: '..DATA_FOLDER..' params: ')
    print(params)
    print_hyperparameters()

    local images_folder = DATA_FOLDER
    local path, modelString
    folder_and_name = get_last_used_model_folder_and_name()
    path = folder_and_name[1]
    modelString = folder_and_name[2]
    local  model = torch.load(path..'/'..modelString)
    if USE_CUDA then
       model = model:cuda()
    else
       model = model:double()
    end

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



-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-use_cuda', false, 'true to use GPU, false (default) for CPU only mode')
cmd:option('-use_continuous', false, 'true to use a continuous action space, false (default) for discrete one (0.5 range actions)')
cmd:option('-data_folder', STATIC_BUTTON_SIMPLEST, 'Possible Datasets to use: staticButtonSimplest, mobileRobot, staticButtonSimplest, simpleData3D, pushingButton3DAugmented, babbling')
cmd:option('-mcd', 0.4, 'Max. cosine distance allowed among actions for priors loss function evaluation (MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)')
cmd:option('-sigma', 0.6, "Sigma: denominator in continuous actions' extra factor (CONTINUOUS_ACTION_SIGMA)")

local params = cmd:parse(arg)

main(params)
