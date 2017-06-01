require 'functions'

images_folder = 'mobileRobot'
list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(images_folder)

print("list_txt_button",list_txt_button)
all_button = {}

for num_line, seq_str in ipairs(list_txt_button) do
   local t,_ = tensorFromTxt(seq_str)

   for num_button=1,t:size(1)-1 do
         all_button[#all_button+1] = t[num_button][1]
   end
end

outStr = ''
for num_line=1,#all_button do
   outStr = outStr..all_button[num_line]..' \n'
end

f = io.open('allRewards.txt', 'w')
f:write(outStr)
f:close()
