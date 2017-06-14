require 'lfs'
require 'torch'
require 'image'
require 'nn'
require 'nngraph'

require 'const'
require 'functions'

if USE_CUDA then
   require 'cunn'
   require 'cudnn'
end

local imagesFolder = DATA_FOLDER
local path, model_string
-- Last model is a file where the name of the last model computed is saved
-- this way, you just have to launch the programm without specifying anything,
-- and it will load the good model

if arg[2] then
   path = arg[1]
   model_string = arg[2]
   DIFFERENT_FORMAT = false
   IM_LENGTH = 200
   IM_HEIGHT = 200
   
elseif file_exists(LAST_MODEL_FILE) then
   f = io.open(LAST_MODEL_FILE,'r')
   path = f:read()
   model_string = f:read()
   print('MODEL USED (last model logged in '..LAST_MODEL_FILE..') : '..model_string)
   f:close()
else
   error(LAST_MODEL_FILE.." should exist")
end

local  model = torch.load(path..'/'..model_string)
if USE_CUDA then
   model = model:cuda()
else
   model = model:double()
end

outStr = ''
tempSeq = {}

--returns the representation of the image (a tensor of size {1xDIM})
function represent_all_images(imagesFolder)
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
               else
                  img = getImageFormated(fullImagesPath)
               end

               img = img:double():reshape(1,IM_CHANNEL,IM_LENGTH,IM_HEIGHT)
               
               if USE_CUDA then
                  img = img:cuda()
               end

               repr = model:forward(img)
               -- print ('img and repr')
               -- print (#img)
               -- print(repr)
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
end

tempSeq = represent_all_images(imagesFolder)

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
