require 'functions'

local function main(params)
    print("\n\n>> create_plotStates_file_for_all_seq: Creating all states file for NN-Quantitative Criterion plot. ") --
    set_hyperparams(params)  --only relevant params are set (cuda usage in this case only)
    print_hyperparameters(true, 'create_plotStates_file_for_all_seq.lua Hyperparams')

    local images_folder = DATA_FOLDER
    print('In DATA_FOLDER: '..images_folder)--..' params: ')    print(params)


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

    f = io.open('allStatesGT.txt', 'w')-- for last model run only, but to avoid dataset related errors, use the next file below:
    f:write(outStr)
    f = io.open('allStatesGT_'..images_folder..'.txt', 'w') -- for evaluation purposes efficiency
    f:write(outStr)
    f:close()
end


-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-use_cuda', false, 'true to use GPU, false (default) for CPU only mode')
cmd:option('-use_continuous', false, 'true to use a continuous action space, false (default) for discrete one (0.5 range actions)')
cmd:option('-data_folder', MOBILE_ROBOT, 'Possible Datasets to use: staticButtonSimplest, mobileRobot, staticButtonSimplest, simpleData3D, pushingButton3DAugmented, babbling')
cmd:option('-mcd', 0.5, 'Max. cosine distance allowed among actions for priors loss function evaluation (MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)')
cmd:option('-sigma', 0.1, "Sigma: denominator in continuous actions' extra factor (CONTINUOUS_ACTION_SIGMA)")

local params = cmd:parse(arg)

main(params)
