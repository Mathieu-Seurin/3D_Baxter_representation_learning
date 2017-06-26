require 'functions'


local function main(params)
    print("\n\n>> create_all_rewards.lua: Creating all rewards for plot")
    set_hyperparams(params)
    local images_folder = DATA_FOLDER
    print('In DATA_FOLDER: '..images_folder..' params: ')
    print(params)

    list_folders_images, list_txt_action,list_txt_button, list_txt_state= Get_HeadCamera_View_Files(images_folder)
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
