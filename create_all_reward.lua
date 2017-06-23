require 'functions'


local function main(params)
    set_hyperparams(params)--set_basic_hyperparams(params)

    local images_folder = DATA_FOLDER
    -- if DATA_FOLDER then
    --     local images_folder = DATA_FOLDER
    -- else --when not using command line to set hyperparameters and calling this script in a pipeline
    --     local images_folder = get_data_folder_from_model_name(get_last_used_model_folder_and_name()[2])
    --     --images_folder = MOBILE_ROBOT --DATA_FOLDER --does not work if we set DATA_FOLDER only on script taking from command line and thus we extract it from the last model trained
    --     --However, I do not know why the constant in const is set for imagesAndReprToTxt (even if I require 'const' here as well, but is is nil when it comes to run this script)
    --     set_minimum_hyperparams_for_dataset(images_folder)
    -- end

    print("\n\n>> create_all_rewards.lua: Creating all rewards for plot. DATA_FOLDER: "..images_folder)
    list_folders_images, list_txt_action,list_txt_button, list_txt_state= Get_HeadCamera_View_Files(images_folder)
    -- print(list_folders_images[1])
    -- print(list_txt_action[1])
    -- print(list_txt_button[1])
    -- print(list_txt_state[1])
    all_button = {}


    for num_line, seq_str in ipairs(list_txt_button) do
        local t,_ = tensorFromTxt(seq_str)
        for num_button=1,t:size(1) do
          all_button[#all_button+1] = t[num_button][REWARD_INDEX]
        end
    end

    outStr = ''
    for num_line=1,#all_button do
       outStr = outStr..all_button[num_line]..' \n'
    end

    f = io.open('allRewards.txt', 'w') -- for last model run
    f:write(outStr)
    f = io.open('allRewards_'..images_folder..'.txt', 'w') -- for evaluation purposes efficiency
    f:write(outStr)
    f:close()
end


-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-use_cuda', false, 'true to use GPU, false (default) for CPU only mode')
cmd:option('-use_continuous', false, 'true to use a continuous action space, false (default) for discrete one (0.5 range actions)')
cmd:option('-data_folder', STATIC_BUTTON_SIMPLEST, 'Possible Datasets to use: staticButtonSimplest, mobileRobot, staticButtonSimplest, simpleData3D, pushingButton3DAugmented, babbling')
cmd:option('-mcd', 0.4, 'Max. cosine distance allowed among actions for priors loss function evaluation (MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)')
cmd:option('-sigma', 0.6, "Sigma: denominator in continuous actions' extra factor (CONTINUOUS_ACTION_SIGMA)")

local params = cmd:parse(arg)

main(params)
