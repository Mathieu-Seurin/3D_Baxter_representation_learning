require 'const'
require 'image'
require 'Get_Images_Set'
tnt = require 'torchnet'
vision = require 'torchnet-vision'

---------------------------------------------------------------------------------------
-- Function :save_model(model,path)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function save_model(model)

   path = LOG_FOLDER..NAME_SAVE
   lfs.mkdir(path)
   file_string = path..'/'..NAME_SAVE..'.t7'

   os.execute("cp hyperparams.lua "..path)

   torch.save(file_string, model)
   print("Saved model at : "..path)

   f = io.open(LAST_MODEL_FILE,'w')
   f:write(path..'\n'..NAME_SAVE..'.t7')
   f:close()
end

---------------------------------------------------------------------------------------
-- Function :getRandomBatchFromSeparateList(batch_size, mode)
-- Input ():
-- Output (): returns the batch with images and the two actions associated to be
-- considered in the computation of the loss funtion based on priors
---------------------------------------------------------------------------------------
function getRandomBatchFromSeparateList(batch_size, mode)

   if mode=="Prop" or mode=="Rep" then
      batch=torch.Tensor(4, batch_size, IM_CHANNEL, IM_LENGTH, IM_HEIGHT)
   else
      batch=torch.Tensor(2, batch_size, IM_CHANNEL, IM_LENGTH, IM_HEIGHT)
   end

   local im1,im2,im3,im4,set

   for i=1, batch_size do

      INDEX1=torch.random(1,NB_SEQUENCES) -- Global only for visualisation purposes
      INDEX2=torch.random(1,NB_SEQUENCES) -- Global only for visualisation purposes

      local data1,data2

      if CAN_HOLD_ALL_SEQ_IN_RAM then
         data1 = ALL_SEQ[INDEX1]
         data2 = ALL_SEQ[INDEX2]
      else
         data1 = load_seq_by_id(INDEX1)
         data2 = load_seq_by_id(INDEX2)
      end

      assert(data1, "Something went wrong while loading data1")
      assert(data2, "Something went wrong while loading data2")

      if mode=="Prop" or mode=="Rep" then
         set = get_two_Prop_Pair(data1.Infos, data2.Infos)
         im1,im2 = data1.images[set.im1], data1.images[set.im2]
         im3,im4 = data2.images[set.im3], data2.images[set.im4]
         batch[1][i]= im1
         batch[2][i]= im2
         batch[3][i]= im3
         batch[4][i]= im4
      elseif mode=="Temp" then
         set=get_one_random_Temp_Set(#data1.images)
         im1,im2 = data1.images[set.im1], data1.images[set.im2]

        --  print("im1",im1:size())
        --  print("im2",im2:size())

         batch[1][i]=im1
         batch[2][i]=im2
      elseif mode=="Caus" then
         set=get_one_random_Caus_Set(data1.Infos, data2.Infos)
         im1,im2,im3,im4 = data1.images[set.im1], data2.images[set.im2], data1.images[set.im3], data2.images[set.im4]

         --The last two are for viz purpose only
         batch[1][i]=im1
         batch[2][i]=im2

         im2,im3 = im3,im2 --I switch them for a better viz, that's all

      else
         print "getRandomBatchFromSeparateList Wrong mode "
      end

      if LOGGING_ACTIONS and mode=='Caus' then

         if LOG_ACTION[INDEX1][set.im1] then
            LOG_ACTION[INDEX1][set.im1] = LOG_ACTION[INDEX1][set.im1]+ 1
         else
            LOG_ACTION[INDEX1][set.im1] = 1
         end

         if LOG_ACTION[INDEX2][set.im2] then
            LOG_ACTION[INDEX2][set.im2] = LOG_ACTION[INDEX2][set.im2]+ 1
         else
            LOG_ACTION[INDEX2][set.im2] = 1
         end

      end

   end

   --Very useful tool to check if prior are coherent
   if VISUALIZE_IMAGES_TAKEN then
      print("MODE :",mode)
      visualize_set(im1,im2,im3,im4)
   end
   return batch, set.act1, set.act2
end

---------------------------------------------------------------------------------------
-- Function :	load_seq_by_id(id)
-- Input (): id of the record file. Loads data of that sequence id, and if it does not exists, it creates the preprocessed data and saves into the PRELOAD_FOLDER
-- Output ():  Returns a Lua Table with the fields:
-- images (e.g., array of 100 float Tensors of 200x200)
-- Infos: 2 indexed arrays, e.g. in mobileData: of 100 values
-- reward (array of 100 indexed rewards)
---------------------------------------------------------------------------------------
function load_seq_by_id(id)
   local string_precomputed_data

   if IS_INCEPTION then
      -- since the model require images to be a 3x299x299
      --and normalize differently, we need to adapt
      string_precomputed_data =
         PRELOAD_FOLDER..'preloaded_'..DATA_FOLDER..'_Seq'..id..'_299.t7'

   else
      string_precomputed_data =
         PRELOAD_FOLDER..'preloaded_'..DATA_FOLDER..'_Seq'..id..'_normalized.t7'
   end

   -- DATA EXISTS
   if file_exists(string_precomputed_data) then
      data = torch.load(string_precomputed_data)
      --print("load_seq_by_id: Data exists in "..string_precomputed_data..".  Loading...")
   else   -- DATA DOESN'T EXIST AT ALL
      print("load_seq_by_id input file DOES NOT exists (input id "..id..") Getting files and saving them to "..string_precomputed_data..' from DATA_FOLDER '..DATA_FOLDER)
      local list_folders_images, list_txt_action,list_txt_button, list_txt_state = Get_HeadCamera_View_Files(DATA_FOLDER)
      --print('Get_HeadCamera_View_Files returned #folders: '..#list_folders_images) --print(list_folders_images)
      if #list_folders_images == 0 then
         error("load_seq_by_id: list_folders_images returned by Get_HeadCamera_View_Files is empty! ",#list_folders_images)
      end
      assert(list_folders_images[id], 'The frame with order id '..id..'  within the record '..string_precomputed_data..' does not correspond to any existing frame. Check the NB_BATCHES parameter for this dataset and adjust it accounting for the average nr of frames per record')
      local list= images_Paths(list_folders_images[id])
      local txt = list_txt_action[id]
      local txt_reward = list_txt_button[id] --nil
      local txt_state = list_txt_state[id]--nil

      data = load_Part_list(list, txt, txt_reward, txt_state)--      print("load_Part_list: ",#data) --for tables, #table returns 0 despite not being empty table.       print (data)
      torch.save(string_precomputed_data, data)
   end
   assert(data, 'Failure in load_seq_by_id: data to be saved is nil')
   return data
end

---------------------------------------------------------------------------------------
-- Function : load_list(list,length,height)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function load_Part_list(list, txt, txt_reward, txt_state)

   assert(list, "list not found")
   assert(txt, "Txt not found")
   assert(txt_state, "Txt state not found")
   assert(txt_reward, "Txt reward not found")

   local all_images={}
   local Infos = getInfos(txt,txt_reward,txt_state)
   -- print('list size: '..#list)
   -- print('Infos[1] size: '..#Infos[1])
   -- print ('Infos size: '..#Infos)
   -- print ('#(Infos.reward): '..#(Infos.reward))-- 11, 99  2  99
   assert(#Infos[1]==#list)   -- assert(#(Infos.reward)==#list)
   assert(#(Infos.reward)== #Infos[1])

   if IS_INCEPTION then
      augmentation = tnt.transform.compose{
         vision.image.transformimage.colorNormalize{
            mean = MEAN_INCEPTION, std  = STD_INCEPTION
         },
         function(img) return img:float() end
      }
      for i=1, #(Infos[1]) do
         im = image.scale(image.load(list[i],3,'float'), IM_LENGTH, IM_HEIGHT)
         table.insert(all_images, augmentation(im))
      end

   else
      for i=1, #(Infos[1]) do
         table.insert(all_images, getImageFormated(list[i]))
      end
   end

   return {images=all_images, Infos=Infos}
end

function getInfos(txt,txt_reward,txt_state)

   local Infos={}
   for dim=1,DIMENSION_IN do
      Infos[dim] = {}
   end
   Infos.reward = {}

   local reward_index= REWARD_INDEX

   local tensor_state, label=tensorFromTxt(txt_state)

   local tensor, label=tensorFromTxt(txt)
   local tensor_reward, label=tensorFromTxt(txt_reward)
   local there_is_reward=false

   for i=1,tensor_reward:size(1) do

      local last_pos = {}
      for dim=1,#INDEX_TABLE do
         id_of_dim_in_tensor = INDEX_TABLE[dim]
         local value = tensor_state[i][id_of_dim_in_tensor]
         table.insert(Infos[dim],value)
         table.insert(last_pos, value) -- For out_of_bound func
      end

      local reward = tensor_reward[i][reward_index]
      if reward ~=0 then
         there_is_reward=true
         table.insert(Infos.reward, reward)
      elseif is_out_of_bound(last_pos) then
         there_is_reward=true
         table.insert(Infos.reward,-1)
      else
         table.insert(Infos.reward,0)
      end
      --print(tensor_reward[i][reward_index])
   end
   --THIS IS ALWAYS THE CASE IF WE WANT TO USE CAUSALITY PRIORS. TODO: create synthetic second value reward or do notn apply causality prior (see PRIORS_TO_APPLY in const.lua)
   if DATA_FOLDER ~= BABBLING then
      assert(there_is_reward,"Reward different than 0 (i.e. min 2 different values of reward) are needed in a sequence...")
      -- else
      --     print('Causality prior will be ignored for dataset '..BABBLING)
   end
   return Infos
end

---------------------------------------------------------------------------------------
-- Function :	applies_prior(list_prior,prior)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function applying_prior(priors_used, prior)
   return list_contains_element(priors_used, prior)
end



---------------------------------------------------------------------------------------
-- Function :	list_contains_element(list, element)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function list_contains_element(list, element)
   if #list ~=0 then
      for i=1, #list do
         if list[i] == element then return true end
      end
   end
   return false
end

---------------------------------------------------------------------------------------
-- Function :	Get_Folder_Name(Log_Folder,Prior_Used)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Get_Folder_Name(Log_Folder,list_prior)
   name=''
   if #list_prior~=0 then
      if #list_prior==1 then
         name=list_prior[1].."_Only"
      elseif #list_prior==4 then
         name='Everything'
      else
         name=list_prior[1]
         for i=2, #list_prior do
            name=name..'_'..list_prior[i]
         end
      end
   end
   return Log_Folder..name..'/'
end

---------------------------------------------------------------------------------------
-- Function : load_list(list,length,height)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function load_Part_list(list, txt, txt_reward, txt_state)

   assert(list, "list not found")
   assert(txt, "Txt not found")
   assert(txt_state, "Txt state not found")
   assert(txt_reward, "Txt reward not found")

   local im={}
   local Infos = getInfos(txt,txt_reward,txt_state)
   -- print('list size: '..#list)
   -- print('Infos[1] size: '..#Infos[1])
   -- print ('Infos size: '..#Infos)
   -- print ('#(Infos.reward): '..#(Infos.reward))-- 11, 99  2  99
   assert(#Infos[1]==#list)   -- assert(#(Infos.reward)==#list)
   assert(#(Infos.reward)== #Infos[1])
   for i=1, #(Infos[1]) do
      table.insert(im, getImageFormated(list[i]))
   end

   return {images=im, Infos=Infos}
end

function getInfos(txt,txt_reward,txt_state)

   local Infos={}
   for dim=1,DIMENSION_IN do
      Infos[dim] = {}
   end
   Infos.reward = {}

   local reward_index= REWARD_INDEX

   local tensor_state, label=tensorFromTxt(txt_state)

   local tensor, label=tensorFromTxt(txt)
   local tensor_reward, label=tensorFromTxt(txt_reward)
   local there_is_reward=false

   for i=1,tensor_reward:size(1) do

      local last_pos = {}
      for dim=1,#INDEX_TABLE do
         id_of_dim_in_tensor = INDEX_TABLE[dim]
         local value = tensor_state[i][id_of_dim_in_tensor]
         table.insert(Infos[dim],value)
         table.insert(last_pos, value) -- For out_of_bound func
      end

      local reward = tensor_reward[i][reward_index]
      if reward ~=0 then
         there_is_reward=true
         table.insert(Infos.reward, reward)
      elseif is_out_of_bound(last_pos) then
         there_is_reward=true
         table.insert(Infos.reward,-1)
      else
         table.insert(Infos.reward,0)
      end
      --print(tensor_reward[i][reward_index])
   end
   --THIS IS ALWAYS THE CASE IF WE WANT TO USE CAUSALITY PRIORS. TODO: create synthetic second value reward or do notn apply causality prior (see PRIORS_TO_APPLY in const.lua)
   if DATA_FOLDER ~= BABBLING then
      assert(there_is_reward,"Reward different than 0 (i.e. min 2 different values of reward) are needed in a sequence...")
      -- else
      --     print('Causality prior will be ignored for dataset '..BABBLING)
   end
   return Infos
end

function scaleAndCrop(img)
   --No random cropping at the moment, but might me useful in the future.

   local format=IM_LENGTH.."x"..IM_HEIGHT
   local imgAfter=image.scale(img,format)

   if VISUALIZE_IMAGE_CROP then
      dim1_before = img:size(1)
      dim2_before = img:size(2)
      dim3_before = img:size(3)

      dim1_after = imgAfter:size(1)
      dim2_after = imgAfter:size(2)
      dim3_after = imgAfter:size(3)

      imgAfterPadded =torch.zeros(dim1_before,dim2_before, dim3_before)
      imgAfterPadded[{{1,dim1_after},{1,dim2_after},{1,dim3_after}}] =
         imgAfter

      local imgMerge = image.toDisplayTensor({img,imgAfterPadded})
      print("Before and After scale")
      image.display{image=imgMerge,win=WINDOW}
      io.read()
   end

   return imgAfter
end

--list_positions is a table with (DIM, normally 3) coordinate positions
function is_out_of_bound(list_positions)
   -- For each dimension you check if the value is inside
   -- barrier fix by MIN_TABLE and MAX_TABLE
   for dim=1,#list_positions do
      if list_positions[dim] < MIN_TABLE[dim] or list_positions[dim] > MAX_TABLE[dim] then
         return true
      end
   end
   return false
end

function calculate_mean_and_std()
   -- This function can work on its own
   -- Just need the global variable DATA_FOLDER to be set

   print("Calculating Mean and Std for all images in ", DATA_FOLDER)

   local imagesFolder = DATA_FOLDER

   local mean = {torch.zeros(IM_LENGTH,IM_HEIGHT),torch.zeros(IM_LENGTH,IM_HEIGHT),torch.zeros(IM_LENGTH,IM_HEIGHT)}
   local std = {torch.zeros(IM_LENGTH,IM_HEIGHT),torch.zeros(IM_LENGTH,IM_HEIGHT),torch.zeros(IM_LENGTH,IM_HEIGHT)}
   local totImg = 0

   for i=1,3 do
      mean[i] = mean[i]:float()
      std[i] = std[i]:float()
   end

   for seqStr in lfs.dir(imagesFolder) do
      if string.find(seqStr,'record') then
         print("seqStr",seqStr)
         local imagesPath = imagesFolder..'/'..seqStr..'/'..SUB_DIR_IMAGE
         for imageStr in lfs.dir(imagesPath) do
            if string.find(imageStr,'jpg') then
               totImg = totImg + 1
               local fullImagesPath = imagesPath..'/'..imageStr
               local img=image.load(fullImagesPath,3,'float')
               img = scaleAndCrop(img)

               mean[1] = mean[1]:add(img[{1,{},{}}])
               mean[2] = mean[2]:add(img[{2,{},{}}])
               mean[3] = mean[3]:add(img[{3,{},{}}])
            end
         end
      end
   end

   mean[1] = mean[1] / totImg
   mean[2] = mean[2] / totImg
   mean[3] = mean[3] / totImg

   for seqStr in lfs.dir(imagesFolder) do
      if string.find(seqStr,'record') then
         local imagesPath = imagesFolder..'/'..seqStr..'/'..SUB_DIR_IMAGE
         for imageStr in lfs.dir(imagesPath) do
            if string.find(imageStr,'jpg') then
               local fullImagesPath = imagesPath..'/'..imageStr
               local img=image.load(fullImagesPath,3,'float')
               img = scaleAndCrop(img)
               std[1] = std[1]:add(torch.pow(img[{1,{},{}}]-mean[1],2))
               std[2] = std[2]:add(torch.pow(img[{2,{},{}}]-mean[2],2))
               std[3] = std[3]:add(torch.pow(img[{3,{},{}}]-mean[3],2))
            end
         end
      end
   end

   std[1] = torch.sqrt(std[1] / totImg)
   std[2] = torch.sqrt(std[2] / totImg)
   std[3] = torch.sqrt(std[3] / totImg)

   im_mean = torch.zeros(3,200,200)
   im_std = torch.zeros(3,200,200)

   for i=1,3 do
      im_mean[i] = mean[i]
      im_std[i] = std[i]
   end

   im_mean = im_mean:float()
   im_std = im_std:float()
   torch.save(STRING_MEAN_AND_STD_FILE,{mean=im_mean,std=im_std})
   return im_mean,im_std
end


function normalize(im)

   local meanStd, mean, std, im_norm, imgMerge

   -- print("im1")
   -- image.display{image=im, win=WINDOW}
   -- io.read()

   if file_exists(STRING_MEAN_AND_STD_FILE) then
      meanStd = torch.load(STRING_MEAN_AND_STD_FILE)
      mean = meanStd.mean
      std = meanStd.std
   else
      mean, std = calculate_mean_and_std()
   end

   im_norm = torch.add(im,-mean)
   --im_norm = torch.cdiv(im_norm, std)

   -- print("im2",im[1][1][1])
   -- image.display{image=im, win=WINDOW}
   -- io.read()

   if VISUALIZE_MEAN_STD then
      --imgMerge = image.toDisplayTensor({mean,std,im,im_norm})
      imgMerge = image.toDisplayTensor({mean,im,im_norm})

      print("Mean, im, im_norm")
      image.display{image=imgMerge, win=WINDOW}
      io.read()
   end

   return im_norm
end

function getImageFormated(im)
   if im=='' or im==nil then error("im is nil, this is not an image") end
   local img=image.load(im,3,'float')
   img = scaleAndCrop(img)
   if NORMALIZE_IMAGE then
      img = normalize(img)
   end
   return img
end


function file_exists(name)
   --tests whether the file can be opened for reading
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function folder_exists(strFolderName)
   local fileHandle, strError = io.open(strFolderName.."\\*.*","r")
   if fileHandle ~= nil then
      io.close(fileHandle)
      return true
   else
      if string.match(strError,"No such file or directory") then
         return false
      else
         return true
      end
   end
end

function visualize_image_from_seq_id(seq_id,image_id1,image_id2, another_window)
   local data = load_seq_by_id(seq_id).images
   local image1

   if image_id2 then
      image1 = data[image_id1]
      local image2 = data[image_id2]
      local imgMerge = image.toDisplayTensor({image1,image2})

      if another_window then
         image.display{image=imgMerge,win=a}
      else
         image.display{image=imgMerge,win=WINDOW}
      end

   else
      image1 = data[image_id1]
      image.display{image=image1,win=WINDOW}
   end
end

function visualize_set(im1,im2,im3,im4)

   if im3 then --Caus or temp
      imgMerge = image.toDisplayTensor({im1,im2,im3,im4})
      image.display{image=imgMerge, win=WINDOW}
   else --Rep or prop
      imgMerge = image.toDisplayTensor({im1,im2})
      image.display{image=imgMerge, win=WINDOW}
   end
   io.read()
end



---------------------------------------------------------------------------------------
-- Function : actions_difference(action1, action2).
-- Input: two vectors representing 2 actions (because actions represent the movement of the arm from one position state to the next one)
-- Output (): Returns a double indicating the MSE (Euclidean distance for our vectors) among actions
---------------------------------------------------------------------------------------
function actions_difference(action1, action2)
  --for each dim, check that the magnitude of the action is close
  return MSE(action1, action2) --TODO change to cosDistance?
  --return CosineDistance(action1, action2)
end

---------------------------------------------------------------------------------------
-- Function : MSE(vec1, vec2, dim)
-- Output (): Returns a double indicating the Euclidean distance among the two points of dimension dim
---------------------------------------------------------------------------------------
function MSE(vec1, vec2)
  local mse = 0
  --for each dimension, add the magnitude of the difference
  for dim=1, #(vec1[1]) do
     mse = mse + (math.pow(arrondit(vec1[dim]) - arrondit(vec2[dim]), 2))
  end
  print ('mse v1 and 2: ')
  print(vec1)
  print('arrondit')
  print(vec1[dim])
  print(arrondit(vec1[dim]))
  print(vec2)
  if USE_CUDA then
    print(vec2[dim]:cudaHalf())
    print('half precision')
  end
  print(math.sqrt(mse))
  print("MSE for vectors size ", #(vec1[1]))
  return math.sqrt(mse)
end


---------------------------------------------------------------------------------------
-- Function : action_vectors_are_similar_enough(action1, action2)
-- Input (): 2 tables of dim DIMENSION_IN
-- Cos(a,b) can be in [-1, 1] (two vectors
-- at 90° have a similarity of 0, and two vectors diametrically opposed have a similarity of -1,
-- independent of their magnitude.
---------------------------------------------------------------------------------------
function cosineDistance(table1, table2)
  -- Returns 1- cos(table1, table2)
  cos = nn.CosineDistance()
  --return cos:forward({t1, t2})-- input is Tensors
  return cos:forward({table2tensor(table1), table2tensor(table2)})
end
