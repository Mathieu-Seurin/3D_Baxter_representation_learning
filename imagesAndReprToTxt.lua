require 'lfs'
require 'torch'
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'

require 'const'
require 'functions'

local imagesFolder = DATA_FOLDER
local path, modelString
-- Last model is a file where the name of the last model computed is saved
-- this way, you just have to launch the programm without specifying anything,
-- and it will load the good model

if file_exists(LAST_MODEL_FILE) then
   f = io.open(LAST_MODEL_FILE,'r')
   path = f:read()
   modelString = f:read()
   print('MODEL USED (last model logged in ) : '..modelString)
   f:close()
else
   error(LAST_MODEL_FILE.." should exist")
end

local  model = torch.load(path..'/'..modelString)
if USE_CUDA then
  model = model:cuda()
else
  model = model:double()
end

outStr = ''

tempSeq = {}
for dir_seq_str in lfs.dir(imagesFolder) do
   if string.find(dir_seq_str,'record') then
      print("Sequence (data record id): ",dir_seq_str)
      local imagesPath = imagesFolder..'/'..dir_seq_str..'/'..SUB_DIR_IMAGE
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
            -- print ('img and repr')
            -- print (#img)
            -- print(repr)
            for i=1,repr:size(2) do
               reprStr = reprStr..repr[{1,i}]..' '
               --print (reprStr)
            end
            tempSeq[#tempSeq+1] = {fullImagesPath, fullImagesPath..' '..reprStr}
         end
      end
   end


end

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
