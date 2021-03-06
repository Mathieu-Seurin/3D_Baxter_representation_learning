 require 'const'
--require 'functions' -- for cosineDistance

---------------------------------------------------------------------------------------
-- Function : images_Paths(path)  #TODO remove to avoid conflict with 1D
-- Input (Path): path of a Folder which contains jpg images
-- Output : list of the jpg files path
---------------------------------------------------------------------------------------
function images_Paths(folder_containing_jpgs)
   assert(folder_containing_jpgs, 'images_Paths got input folder name nil')
   local listImage={}
   --print('images_Paths: ', folder_containing_jpgs)
   --folder_containing_jpgs="./data_baxter" -- TODO: make it work by passing it as a parameter
   --print (folder_containing_jpgs)
   for file in paths.files(folder_containing_jpgs) do
      --print('getting image path:  '..file)
      -- We only load files that match the extension
      if file:find('jpg' .. '$') then
         -- and insert the ones we care about in our table
         table.insert(listImage, paths.concat(folder_containing_jpgs,file))
         --print('Inserted image :  '..paths.concat(folder_containing_jpgs,file))
      end
   end
   table.sort(listImage)
   --print('Loaded images from Path: '..folder_containing_jpgs)
   return listImage
end

function Get_Folders(Path, including, excluding,list)
   local list=list or {}
   local incl=including or ""
   local excl=excluding or "uyfouhjbhytfoughl" -- random motif

   for file in paths.files(Path) do
      -- We only load files that match 2016 because we know that there are the folder we are interested in
      if file:find(incl) and (not file:find(excl)) then
         -- and insert the ones we care about in our table
         --print('Get_Folders '..Path..' found search pattern: '..incl..' in filename: '..paths.concat(Path,file))
         table.insert(list, paths.concat(Path,file))
         --  else
         -- 	 print('Get_Folders '..Path..' did not find pattern: '..incl..' Check the structure of your data folders')
      end
   end
   return list
end

function Get_HeadCamera_View_Files(Path)
   local use_simulate_images=use_simulate_images or false
   local Paths=Get_Folders(Path,'record')
   list_folder={}
   list_txt_button={}
   list_txt_action={}
   list_txt_state={}
   txt_button_position = {}
   table.insert(txt_button_position, get_path_to_text_files(Paths[1], FILENAME_FOR_BUTTON_POSITION))
   for i=1, #Paths do
      list_folder = Get_Folders(Paths[i],SUB_DIR_IMAGE,'txt',list_folder)
      table.insert(list_txt_button, get_path_to_text_files(Paths[i],FILENAME_FOR_REWARD))
      table.insert(list_txt_action, get_path_to_text_files(Paths[i],FILENAME_FOR_ACTION))
      table.insert(list_txt_state, get_path_to_text_files(Paths[i],FILENAME_FOR_STATE))
   end
   table.sort(list_txt_button) -- file recorded_button_is_pressed.txt
   table.sort(list_txt_action) --fileoss fix
   table.sort(list_txt_state)--recroded_robot_libm_left_endpoint_state  -- for the hand position
   table.sort(list_folder) --recorded_cameras_head_camera_2_image_compressed
   --print(FILENAME_FOR_REWARD,FILENAME_FOR_ACTION,FILENAME_FOR_STATE)
   print('Get_HeadCamera_View_Files: ')
   print(Paths)
   print(get_path_to_text_files(Paths[1], FILENAME_FOR_BUTTON_POSITION))
   return list_folder, list_txt_action, list_txt_button, list_txt_state, txt_button_position
end

---------------------------------------------------------------------------------------
-- Function : Finds .txt files within the records folders including the pattern including and '.txt'
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function get_path_to_text_files(Path, including, excluding)
   local incl=including or ""
   local excl=excluding or "uyfouhjbhytfoughl" -- random motif
   local txt=nil

   for file in paths.files(Path) do
      -- We only load files that match the 'including' pattern because we know that there are the folder we are interested in
      -- super confusing function: file.find and its parameters including
      --if file:find(incl) and (not file:find(excl)) then--if file:find(incl.. '$') and (not file:find(excl)) then --file:find(incl..'.txt' .. '$') then
      if string.find(file, incl) then -- and string.find(file, '.txt')  if file:find(including..'.txt' .. '$') then  --original that weirdly works being incl  nil: if file:find(including..'.txt' .. '$') then
          --print('found path...'..paths.concat(Path,file))
          txt=paths.concat(Path,file) ---TODO: return as soon as we find one, or return a list of all files that match the search criteria
      end
   end
   return txt
end

---------------------------------------------------------------------------------------
-- Function : tensorFromTxt(path)
-- Input (path) : path of a txt file which contain position of the robot
-- Output (torch.Tensor(data)): tensor with all the joint values (col: joint, line : indice)
-- Output (labels):  name of the joint
---------------------------------------------------------------------------------------
function tensorFromTxt(path)
   local data, raw = {}, {}
   local rawCounter, columnCounter = 0, 0
   local nbFields, labels, _line = nil, nil, nil
   --print('tensorFromTxt path:',path)
   for line in io.lines(path)  do   ---reads each line in the .txt data file
      local comment = false
      if line:sub(1,1)=='#' then
         comment = true
         line = line:sub(2)
      end
      rawCounter = rawCounter +1
      columnCounter=0
      raw = {}
      for value in line:gmatch'%S+' do
         columnCounter = columnCounter+1
         raw[columnCounter] = tonumber(value)
      end

      -- we check that every row contains the same number of data
      if rawCounter==1 then
         nbFields = columnCounter
      elseif columnCounter ~= nbFields then
         error("data dimension for " .. rawCounter .. "the sample is not consistent with previous samples'")
      end

      if comment then labels = raw else table.insert(data,raw) end
   end
   return torch.Tensor(data), labels
end

--============== Tools to get action from the state ===========
--=============================================================
function action_amplitude(infos,id1, id2)
   local action = {}

   for dim=1,DIMENSION_IN do
      action[dim] = infos[dim][id1] - infos[dim][id2]
   end
   return action
end

function is_same_action(action1,action2)
   local same_action = true
   --for each dim, you check that the magnitude of the action is close
   for dim=1,DIMENSION_IN do
      same_action = same_action and arrondit(action1[dim] - action2[dim])==0
   end
   return same_action
end

---------------------------------------------------------------------------------------
-- Function : get_one_random_Temp_Set(list_im)
-- Input (list_lenght) : lenght of the list of images
-- Output : 2 indices of images which are neightboor in the list (and in time)
---------------------------------------------------------------------------------------
function get_one_random_Temp_Set(list_lenght)
   index = torch.random(1,list_lenght-1)
   return {im1= index, im2=index+1}
end

function get_one_random_Prop_Set(Infos1)
   return get_two_Prop_Pair(Infos1,Infos1)
end

---------------------------------------------------------------------------------------
-- Function : get_two_Prop_Pair(txt1, txt2,use_simulate_images)
-- Input (txt1) : path of the file of the first list of joint
-- Input (txt2) : path of the file of the second list of joint
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images (we need this information because the data is not formated exactly the same in the txt file depending on the origin of images)
-- Output : structure with 4 indices which represente a quadruplet (2 Pair of images from 2 different list) for Traininng with prop prior. The variation of joint for on pair should be the same as the variation for the second
-- Output : structure with 4 s which represente a quadruplet (2 Pair of images from 2 different list) for Traininng with prop prior.
-- Returns a Lua table with 4 images and the 2 actions derived from the states
-- {
--   im3 : 19
--   im2 : 46
--   act2 :
--     {
--       1 : 0.051146118604
--       2 : -0.04237182035
--       3 : 0.0452409758369
--     }
--   act1 :
--     {
--       1 : 0.045235814614
--       2 : -0.051011220414
--       3 : 0.0505784082665
--     }
--   im4 : 20
--   im1 : 45
-- }
-- The variation of joint for one pair should be close enough (<CLOSE_ENOUGH_PRECISION_THRESHOLD) in continuous actions, to the variation for the second
---------------------------------------------------------------------------------------
function get_two_Prop_Pair(Infos1, Infos2)

   local watchDog=0

   local size1=#Infos1[1]
   local size2=#Infos2[1]

   local vector=torch.randperm(size2-1)

   while watchDog < 100 do
      local id_ref_action_begin=torch.random(1,size1-1)

      if EXTRAPOLATE_ACTION then --Look at const.lua for more details about extrapolate
         repeat id_ref_action_end=torch.random(1,size1) until (id_ref_action_begin ~= id_ref_action_end)
      else
         id_ref_action_end=id_ref_action_begin+1
      end

      action1 = action_amplitude(Infos1,id_ref_action_begin, id_ref_action_end)

      for i=1, size2-1 do
         local id_second_action_begin=vector[i]

         if EXTRAPOLATE_ACTION then --Look at const.lua for more details about extrapolate
            for id_second_action_end in ipairs(torch.totable(torch.randperm(size2))) do
               action2 = action_amplitude(Infos2, id_second_action_begin, id_second_action_end)

               if is_same_action(action1, action2) then
                  return {im1=id_ref_action_begin,im2=id_ref_action_end,im3=id_second_action_begin,im4=id_second_action_end, act1=action1, act2=action2}
               end
            end
         else --USE THE NEXT IMAGE IN THE SEQUENCE
            id_second_action_end=id_second_action_begin+1
            action2 = action_amplitude(Infos2, id_second_action_begin, id_second_action_end)
            if USE_CONTINUOUS then
                if action_vectors_are_similar_enough(action1, action2) then
                    return {im1=id_ref_action_begin,im2=id_ref_action_end,im3=id_second_action_begin,im4=id_second_action_end, act1=action1, act2=action2}
                end
            elseif is_same_action(action1, action2) then --TODO UNIFY TWO IFS, remove this one when continuous works?
               -- print("indices", INDICE1, INDICE2)
               -- print("id_ref_action_begin,id_ref_action_end,id_second_action_begin,id_second_action_end",id_ref_action_begin,id_ref_action_end,id_second_action_begin,id_second_action_end)
               -- print("action1",action1[1],action1[2],action1[3])
               -- print("action2",action2[1],action2[2],action2[3])
               return {im1=id_ref_action_begin,im2=id_ref_action_end,im3=id_second_action_begin,im4=id_second_action_end, act1=action1, act2=action2}
            end
         end
      end
      watchDog=watchDog+1
   end
   error("PROP WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end

---------------------------------------------------------------------------------------
-- Function : get_one_random_Caus_Set(Infos1, Infos2)
-- Input We need to find images representing a starting state, then the same
-- action applied to this state. The same variation of joint or close enough, should lead to a different reward.
-- for instance, we choose as reward the fact of having a joint value = 0  TODO: ???
-- NB : the two states will be took from different lists but the two list can be the same
-- Output : structure with 4 indices which represente a quadruplet (2 Pair of images from 2 different list) for Training with caus prior,
--  and an array of the delta in between actions (the distance in between 2 actions as Euclidean distance)
-- The variation of the joint position for one pair should be close enough
-- (< MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD) in continuous actions, to the variation for the positiono in the second frame
function get_one_random_Caus_Set(Infos1, Infos2)
   local size1=#Infos1[1]
   local size2=#Infos2[1]
   local watchDog=0

   while watchDog<75 do

      repeat
         --Sample an action, whose reward is not 0
         id_ref_action_begin= torch.random(1,size2-1)

         if EXTRAPOLATE_ACTION_CAUS then --Look at const.lua for more details about extrapolate
            repeat id_ref_action_end=torch.random(1,size2) until (id_ref_action_begin ~= id_ref_action_end)
         else
            id_ref_action_end  = id_ref_action_begin+1
         end

         reward1 = Infos2.reward[id_ref_action_end]
      until (reward1~=0)

      action1 = action_amplitude(Infos2, id_ref_action_begin, id_ref_action_end)

      -- Overriding action: Force the action amplitude to be the same, dirty...
      if CLAMP_CAUSALITY and not EXTRAPOLATE_ACTION_CAUS then
         -- WARNING, THIS IS DIRTY, need to do continous prior
         for dim=1,DIMENSION_IN do
            action1[dim]=clamp_causality_prior_value(action1[dim])
         end
      end

      if VISUALIZE_CAUS_IMAGE then
         print("id1",id_ref_action_begin)
         print("id2",id_ref_action_end)
         print("action1",action1[1],action1[2])--,action[3])
         visualize_image_from_seq_id(INDEX2,id_ref_action_begin,id_ref_action_end,true)
         io.read()
      end

      for i=1, size1 do
         id_second_action_begin=torch.random(1,size1-1) --TODO Shouldnt we check id_second_action_begin != id_ref_action_begin?
         --TODO: and also add here CLAMP_CAUSALITY check just as above?

         if EXTRAPOLATE_ACTION_CAUS then --Look at const.lua for more details about extrapolate
            repeat id_second_action_end=torch.random(1,size1) until (id_second_action_begin ~= id_second_action_end)
         else
            id_second_action_end=id_second_action_begin+1
         end

         --if Infos1.reward[id_second_action_begin]==0 and Infos1.reward[id_second_action_end]~=reward1 then
         if Infos1.reward[id_second_action_end]~=reward1 then -- The constraint is softer
            action2 = action_amplitude(Infos1, id_second_action_begin, id_second_action_end)

            --Visualize images taken if you want
            if VISUALIZE_CAUS_IMAGE then
               print("action2",action2[1],action2[2])--,action[3])
               visualize_image_from_seq_id(INDEX1,id_second_action_begin,id_second_action_end)
               print(is_same_action(action1, action2))
               io.read()
            end

            if USE_CONTINUOUS then
               if action_vectors_are_similar_enough(action1, action2) then
                   return {im1=id_second_action_begin,im2=id_ref_action_begin, im3=id_second_action_end, im4=id_ref_action_end, act1=action1, act2=action2}
               end
            elseif is_same_action(action1, action2) then --discrete actions
               return {im1=id_second_action_begin,im2=id_ref_action_begin, im3=id_second_action_end, im4=id_ref_action_end}
            end
         end
      end
      watchDog=watchDog+1
   end
   error("CAUS WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end

function clamp_causality_prior_value(value, prec, action_amplitude)
   -- ======================================================
   -- Selects the next available consecutive action with a given hard-coded action_amplitude
   -- WARNING THIS VERY DIRTY, WE SHOULD DO CONTINOUS PRIOR
   -- INSTEAD OF THIS
   -- ======================================================
   prec = prec or 0.01
   action_amplitude = action_amplitude or 0.05 --An action has an amplitude either of
   --- 0 or 0.05 in the 'simple3D' database (on each axis), see const.lua
   if math.abs(value) < prec then
      value = 0
   else
      value = sign(value)*action_amplitude
   end
   return value
end

function get_one_random_reward_close_set(Infos1, Infos2)

   local size1=#Infos1[1]
   local size2=#Infos2[1]
   local watchDog=0

   while watchDog<50 do
      repeat
         id_ref_state= torch.random(1,size1)
         reward1 = Infos1.reward[id_ref_state]
      until (reward1~=2) --until (reward1==1)
      -- Since all rewards are different from 2
      -- it means that you take the first action you got
      -- I did this because it's easier if you want to group only certain reward
      -- But yea, if you want to group every reward, this code looks silly

      for i=1, size2 do
         id_second_state=torch.random(1,size2)

         if Infos2.reward[id_second_state]==reward1 then
            return {im1=id_ref_state, im2=id_second_state}
         end
      end
      watchDog=watchDog+1
   end

   error("BRING_CLOSER_REWARD WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end

function get_one_random_predictive_reward_set(Infos1, Infos2)

   local size1=#Infos1[1]
   local size2=#Infos2[1]
   local watchDog=0

   while watchDog<50 do
      repeat
         id_ref_state= torch.random(1,size1)
         reward1 = Infos1.reward[id_ref_state]
      until (reward1~=2) --until (reward1==1)
      -- Since all rewards are different from 2
      -- it means that you take the first action you got
      -- I did this because it's easier if you want to group only certain reward
      -- But yea, if you want to group every reward, this code looks silly

      for i=1, size2 do
         id_second_state=torch.random(1,size2)

         if Infos2.reward[id_second_state]==reward1 then
            return {im1=id_ref_state, im2=id_second_state}
         end
      end
      watchDog=watchDog+1
   end

   error("BRING_CLOSER_REWARD WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end

function get_one_fixed_point_set(Infos1, Infos2)

   local function get_coord(info,id)
      coord = torch.Tensor(DIMENSION_IN)
      for dim=1,DIMENSION_IN do
         coord[dim] = info[dim][id]
      end
      return coord
   end

   local function is_same(t1,t2)
      for dim=1,DIMENSION_IN do
         if t1[dim] < t2[dim]-ROUNDING_VALUE_FIX or t1[dim]>t2[dim]+ROUNDING_VALUE_FIX  then
            return false
         end
      end
      return true
   end

   local function look_for_ref(info,fixed_point)
      local id = 0
      local size1=#info[1]

      for _,i in ipairs(torch.totable(torch.randperm(size1))) do
         coord1 = get_coord(info,i)
         if is_same(coord1, fixed_point) then
            id = i
            break
         end
      end
      return id
   end

   local fixed_point_round = torch.Tensor(FIXED_POS)

   id1 = look_for_ref(Infos1,fixed_point_round)
   id2 = look_for_ref(Infos2,fixed_point_round)

   if id1~=0 and id2~=0 then
      return {im1=id1, im2=id2}
   else
      return nil
   end
end
---------------------------------------------------------------------------------------
-- Function : arrondit(value)
-- Input (tensor) :
-- Input (head_pan_index) :
-- Output (tensor):
---------------------------------------------------------------------------------------
function arrondit(value, prec)
   local prec = prec or DEFAULT_PRECISION
   divFactor = 1/prec
   floor=math.floor(value*divFactor)/divFactor
   ceil=math.ceil(value*divFactor)/divFactor
   if math.abs(value-ceil)>math.abs(value-floor) then result=floor
   else result=ceil end
   return result
end

function sign(value)
   if value < 0 then
      return -1
   else
      return 1
   end
end

------------------ CONTINUOUS actions
--Making actions not be the same but close enough for handling continous actions with priors,
-- This method calibrates the use of sigma in the continuous_factor_term
-- function actions_are_close_enough(action1, action2)
--   --print(action1)
--   local close_enough = true
--   --for each dim, check that the magnitude of the action is close (smaller than MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)
--   for dim=1, DIMENSION_IN do
--      close_enough = close_enough and arrondit(action1[dim] - action2[dim]) < MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
--   end
--   print("actions_are_close_enough ")
--   print(close_enough)
--   return close_enough
-- end

function tensor2table(tensor)
    -- Assumes `t1` is a 2-dimensional tensor
    local t2 = {}
    for i=1,t1:size(1) do
      t2[i] = {}
      for j=1,t1:size(2) do
        t2[i][j] = t1[i][j]
      end
      return t2
    end
end

function table2tensor(table)
    --t2 = torch.Tensor(table)--{table})
    -- print('converted to tensor')
    -- print(type(t2))
    -- print(t2)
    return torch.Tensor(table)
end

---------------------------------------------------------------------------------------
-- Function : action_vectors_are_similar_enough(action1, action2)
-- Input (): 2 tables of dim DIMENSION_IN
-- Because actions are vectors (from one position state to the next one), the distance
-- between actions is 1 - cos similarity (act1, act2).
-- Output (): Returns true if the cosDistance is smaller than MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
---------------------------------------------------------------------------------------
function action_vectors_are_similar_enough(action1, action2)
    --    -- examples of action
    --  {{  1 : -1.3799996700925e-09
    --   2 : 0.372556582859
    -- }}
  cosDistance = cosineDistance(action1, action2)
  -- print(action1)
  -- print(action2)
  -- print("action_vectors_are_similar_enough cosDistance:")
  -- print(cosDistance[1]) --[torch.DoubleTensor of size 1]
  -- print(MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)

  if math.abs(cosDistance) < MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD then
      return true
  else
      return false
  end
end
