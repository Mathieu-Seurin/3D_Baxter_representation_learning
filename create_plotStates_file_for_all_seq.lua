require 'functions'

-- function set_basic_hyperparams(params)
--     USE_CUDA = params.use_cuda
--     USE_CONTINUOUS = params.use_continuous  --DATA_FOLDER = params.data_folder
--     if DATA_FOLDER then
--         images_folder = DATA_FOLDER
--     else --when not using command line to set hyperparameters and calling this script in a pipeline
--         images_folder = get_data_folder_from_model_name(get_last_used_model_folder_and_name()[2])
--         --images_folder = MOBILE_ROBOT --DATA_FOLDER --does not work if we set DATA_FOLDER only on script taking from command line and thus we extract it from the last model trained
--         --However, I do not know why the constant in const is set for imagesAndReprToTxt (even if I require 'const' here as well, but is is nil when it comes to run this script)
--     end
--     DATA_FOLDER = images_folder --set_minimum_hyperparams_for_dataset(images_folder)
--     USE_SECOND_GPU = true
--     --===========================================================
--     -- CUDA CONSTANTS
--     --===========================================================
--     if USE_CUDA and USE_SECOND_GPU then
--        cutorch.setDevice(2)
--     end
--     if USE_CUDA then
--         require 'cunn'
--         require 'cudnn'  --If trouble, installing, follow step 6 in https://github.com/jcjohnson/neural-style/blob/master/INSTALL.md
--     end
--     set_dataset_specific_hyperparams(DATA_FOLDER)
-- end

local function main(params)
    set_hyperparams(params)--)set_basic_hyperparams(params)

    local images_folder = DATA_FOLDER

    print("\n\n>> create_plotStates_file_for_all_seq: Creating all states file for NN-Quantitative Criterion plot. DATA_FOLDER: "..images_folder)
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
end


-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-use_cuda', false, 'true to use GPU, false (default) for CPU only mode')
cmd:option('-use_continuous', false, 'true to use a continuous action space, false (default) for discrete one (0.5 range actions)')
cmd:option('-data_folder', MOBILE_ROBOT, 'Possible Datasets to use: staticButtonSimplest, mobileRobot, staticButtonSimplest, simpleData3D, pushingButton3DAugmented, babbling')
cmd:option('-mcd', 0.4, 'Max. cosine distance allowed among actions for priors loss function evaluation (MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)')
cmd:option('-sigma', 0.6, "Sigma: denominator in continuous actions' extra factor (CONTINUOUS_ACTION_SIGMA)")

local params = cmd:parse(arg)

main(params)
