require 'functions'


images_folder = get_data_folder_from_model_name(get_last_used_model_name())
--local images_folder = DATA_FOLDER --does not work if we set DATA_FOLDER only on script taking from command line and thus we extract it from the last model trained
--However, I do not know why the constant in const is set for imagesAndReprToTxt (even if I require 'const' here as well, but is is nil when it comes to run this script)
set_minimum_hyperparams_for_dataset(images_folder)

print("\n\ncreate_plotStates_file_for_all_seq: Creating all states file for NN-Quantitative Criterion plot. DATA_FOLDER: "..images_folder)
list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(images_folder)
print("Reading rewards from file list_txt_button= ",list_txt_button)
print("list_txt_state: ",list_txt_state)

outStr = ''

local seq = 0
local firstElem = 0

all_state = {}

for num_line, seq_str in ipairs(list_txt_state) do
   local t,_ = tensorFromTxt(seq_str)

   for num_state=1,t:size(1) do
      all_state[#all_state+1] = {}
      for dim=1,DIMENSION_IN do
         all_state[#all_state][dim] = t[num_state][INDEX_TABLE[dim]]
      end
   end
end

-- print("all_state",all_state)
-- io.read()
all_path = {}
for dir_seq_str in lfs.dir(images_folder) do
   if string.find(dir_seq_str,'record') then
      local images_path = images_folder..'/'..dir_seq_str..'/'..SUB_DIR_IMAGE
      for image_str in lfs.dir(images_path) do
         if string.find(image_str,'jpg') then
            local fullImagesPath = images_path..'/'..image_str
            all_path[#all_path+1] = {}
            all_path[#all_path][1] = fullImagesPath
         end
      end
   end
end

table.sort(all_path, function (a,b) return a[1] < b[1] end)
assert(#all_path==#all_state,"He fucked up.")

outStr = ''
for num_line=1,#all_path do
   outStr = outStr..all_path[num_line][1]..' '
   for dim=1,DIMENSION_IN do
      outStr = outStr..all_state[num_line][dim]..' '
   end
   outStr = outStr..'\n'
end

f = io.open('allStates.txt', 'w')-- for last model run
f:write(outStr)
f = io.open('allStates_'..images_folder..'.txt', 'w') -- for evaluation purposes efficiency
f:write(outStr)
f:close()
