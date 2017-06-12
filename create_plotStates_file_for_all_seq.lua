require 'functions'


images_folder = DATA_FOLDER --MOBILE_ROBOT
list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(images_folder)
print("Reading rewards from file list_txt_button= ",list_txt_button, ' in DATA_FOLDER: ', DATA_FOLDER)
print("list_txt_state",list_txt_state)

outStr = ''

local seq = 0
local firstElem = 0

all_state = {}

for num_line, seq_str in ipairs(list_txt_state) do
   local t,_ = tensorFromTxt(seq_str)

   for num_state=1,t:size(1) do
       if DIMENSION_IN ==2 then
         all_state[#all_state+1] = {t[num_state][INDEX_TABLE[1]], t[num_state][INDEX_TABLE[2]]}
     elseif DIMENSION_IN == 3 then
         all_state[#all_state+1] = {t[num_state][INDEX_TABLE[1]], t[num_state][INDEX_TABLE[2]], t[num_state][INDEX_TABLE[3]]}
     else
         error('Extend method to handle other than DIMENSION_IN 2 or 3')
     end
   end
end

-- print("all_state",all_state)
-- io.read()


all_path = {}
for dir_seq_str in lfs.dir(images_folder) do
   if string.find(dir_seq_str,'record') then
      print("Sequence : ",dir_seq_str)

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
   outStr = outStr..all_path[num_line][1]..' '..all_state[num_line][1]..' '..all_state[num_line][2]..' \n'
end

f = io.open('allStates.txt', 'w')
f:write(outStr)
f:close()
