require 'lfs'
require 'torch'
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'

require 'const'
require 'functions'

local imagesFolder = DATA_FOLDER
local modelString

local path, modelString
-- Last model is a file where the name of the last model computed is saved
-- this way, you just have to launch the programm without specifying anything,
-- and it will load the good model
if file_exists('lastModel.txt') then
   f = io.open('lastModel.txt','r')
   path = f:read()
   modelString = f:read()
   print('MODEL USED : '..modelString)
   f:close()
else
   error("lastModel.txt should exist")
end

local  model = torch.load(path..'/'..modelString)
if USE_CUDA then
  model = model:cuda()
else
  model = model:double()
end

outStr = ''

for seqStr in lfs.dir(imagesFolder) do
   if string.find(seqStr,'record') then
      print("Sequence : ",seqStr)
      local imagesPath = imagesFolder..'/'..seqStr..'/'..SUB_DIR_IMAGE
      for imageStr in lfs.dir(imagesPath) do
         if string.find(imageStr,'jpg') then
            local fullImagesPath = imagesPath..'/'..imageStr
            local reprStr = ''
            --img = getImageFormated(fullImagesPath):cuda():reshape(1,3,200,200)
            if USE_CUDA then
              img = getImageFormated(fullImagesPath):cuda():reshape(1,3,200,200)
            else
              img = getImageFormated(fullImagesPath):double():reshape(1,3,200,200)
            end
            repr = model:forward(img)
            print ('img and repr')
            print (#img)
            print(repr)
            for i=1,repr:size(2) do
               reprStr = reprStr..repr[{1,i}]..' '
            end
            outStr = outStr..fullImagesPath..' '..reprStr..'\n'

         end
      end
   end
end

file = io.open(path..'/saveImagesAndRepr.txt', 'w')
file:write(outStr)
file:close()
