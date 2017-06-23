--=============================================================
-- Constants file

-- All constants should be located here for now, in CAPS_LOCK
-- Ex : NB_EPOCHS = 10
-- Disclamer there still plenty of constants everywhere
-- Because it wasn't done like this before, so if you still
-- find global variable, that's normal. You can change it
-- and put it here.

-- Hyperparameters are located in a different file (hyperparameters.lua)
--=============================================================
require 'lfs'
require 'hyperparams'

------DEFAULTS (IF NOT COMMAND LINE ARGS ARE PASSED)
USE_CUDA = true
USE_SECOND_GPU = true
USE_CONTINUOUS = true
MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD = 0.4
CONTINUOUS_ACTION_SIGMA = 0.6
DATA_FOLDER = STATIC_BUTTON_SIMPLES --works best!

--torch.manualSeed(100)
--=====================================
--DATA AND LOG FOLDER NAME etc..
--====================================
PRELOAD_FOLDER = 'preload_folder/'
lfs.mkdir(PRELOAD_FOLDER)

LOG_FOLDER = 'Log/'
MODEL_PATH = LOG_FOLDER

--STRING_MEAN_AND_STD_FILE = PRELOAD_FOLDER..'meanStdImages_'..DATA_FOLDER..'.t7'
LEARNED_REPRESENTATIONS_FILE = "saveImagesAndRepr.txt"
LAST_MODEL_FILE = 'lastModel.txt'
GLOBAL_SCORE_LOG_FILE = 'globalScoreLog.csv'
MODELS_CONFIG_LOG_FILE  = 'modelsConfigLog.csv'
--
-- now = os.date("*t")
-- _, architecture_name = MODEL_ARCHITECTURE_FILE:match("(.+)/(.+)") --architecture_name, _ = split(architecture_name, ".")
--print('Architecture name: '..architecture_name)

-- function addLeadingZero(number)
--     -- Returns a string with a leading zero of the number if the number has only one digit (for model logging and sorting purposes)
--     if number >= 0 and number <= 9 then
--         return "0" .. number
--     else
--         return tostring(number)
--     end
-- end

-- if USE_CONTINUOUS then
--     --DAY = 'Y'..now.year..'_D'..now.day..'_M'..now.month..'_H'..now.hour..'M'..now.min..'S'..now.sec..'_'..DATA_FOLDER..'_'..architecture_name..'_cont'..'_MCD0_'..(MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD*10)..'_S0_'..(CONTINUOUS_ACTION_SIGMA*10)
--     DAY = 'Y'..now.year..'_D'..addLeadingZero(now.day)..'_M'..addLeadingZero(now.month)..'_H'..addLeadingZero(now.hour)..'M'..addLeadingZero(now.min)..'S'..addLeadingZero(now.sec)..'_'..DATA_FOLDER..'_'..architecture_name..'_cont'..'_MCD0_'..(MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD*10)..'_S0_'..(CONTINUOUS_ACTION_SIGMA*10)
-- else
--     --DAY = 'Y'..now.year..'_D'..now.day..'_M'..now.month..'_H'..now.hour..'M'..now.min..'S'..now.sec..'_'..DATA_FOLDER..'_'..architecture_name
--     DAY = 'Y'..now.year..'_D'..addLeadingZero(now.day)..'_M'..addLeadingZero(now.month)..'_H'..addLeadingZero(now.hour)..'M'..addLeadingZero(now.min)..'S'..addLeadingZero(now.sec)..'_'..DATA_FOLDER..'_'..architecture_name
-- end
--
-- NAME_SAVE= 'model'..DAY
-- SAVED_MODEL_PATH = LOG_FOLDER..NAME_SAVE

RELOAD_MODEL = false

--===========================================================
-- VISUALIZATION TOOL
-- if you want to visualize images, use 'qlua' instead of 'th'
--===========================================================
VISUALIZE_IMAGES_TAKEN = false
VISUALIZE_CAUS_IMAGE = false
VISUALIZE_IMAGE_CROP = false
VISUALIZE_MEAN_STD = false
VISUALIZE_AE = false

--
-- if VISUALIZE_IMAGES_TAKEN or VISUALIZE_CAUS_IMAGE or VISUALIZE_IMAGE_CROP or VISUALIZE_MEAN_STD or VISUALIZE_AE then
--    --Creepy, but need a placeholder, to prevent many window to pop
--    WINDOW = image.display(image.lena())
-- end
--
-- LOGGING_ACTIONS = false
--
-- IS_INCEPTION = string.find(MODEL_ARCHITECTURE_FILE, 'inception')
-- -- since the model require images to be a 3x299x299, and normalize differently, we need to adapt
-- IS_RESNET = string.find(MODEL_ARCHITECTURE_FILE, 'resnet')
--
-- DIFFERENT_FORMAT = IS_INCEPTION or IS_RESNET
--
--
-- if IS_INCEPTION then
--    IM_LENGTH = 299
--    IM_HEIGHT = 299
--    MEAN_MODEL = torch.ones(3):double()*0.5
--    STD_MODEL = torch.ones(3):double()*0.5
--
-- elseif IS_RESNET then
--    IM_LENGTH = 224
--    IM_HEIGHT = 224
--    MEAN_MODEL = torch.DoubleTensor({ 0.485, 0.456, 0.406 })
--    STD_MODEL = torch.DoubleTensor({ 0.229, 0.224, 0.225 })
--
-- else
--    IM_LENGTH = 200
--    IM_HEIGHT = 200
-- end
--
-- print ("DIFFERENT_FORMAT")
-- print(DIFFERENT_FORMAT)
-- print ("STD_MODEL")
-- print(STD_MODEL)
-- print(IS_INCEPTION)
-- print(IS_RESNET)
-- print(MODEL_ARCHITECTURE_FILE)
--

IM_CHANNEL = 3 --image channels (RGB)
ACTION_AMPLITUDE = 0.01
--================================================
-- dataFolder specific constants : filename, dim_in, indexes in state file etc...
--================================================
CLAMP_CAUSALITY = false--cant add to functions because it creates an import loop

MIN_TABLE = {-10000,-10000} -- for x,y
MAX_TABLE = {10000,10000} -- for x,y

DIMENSION_IN = 2
DIMENSION_OUT = 2  --worked just as well as 4 output dimensions
REWARD_INDEX = 1  --3 reward values: -1, 0, 10
INDEX_TABLE = {1,2} --column index for coordinate in state file (respectively x,y)

DEFAULT_PRECISION = 0.1
FILENAME_FOR_ACTION = "recorded_robot_action.txt" --not used at all, we use state file, and compute the action with it (contains dx, dy)
FILENAME_FOR_STATE = "recorded_robot_state.txt"
FILENAME_FOR_REWARD = "recorded_robot_reward.txt"

-- WARNING : If you change the folder (top, pano, front)
-- do rm preload_folder/* because the images won't be good
SUB_DIR_IMAGE = 'recorded_camera_top'
AVG_FRAMES_PER_RECORD = 90

--===============================================
PRIORS_CONFIGS_TO_APPLY ={{"Prop","Temp","Caus","Rep"}}
FILE_PATTERN_TO_EXCLUDE = 'deltas'
CAN_HOLD_ALL_SEQ_IN_RAM = false
-- ====================================================
--DATASET DEPENDENT settings to be set below
STRING_MEAN_AND_STD_FILE =''
NAME_SAVE= ''
SAVED_MODEL_PATH = ''
--MODEL_ARCHITECTURE_FILE =''
WINDOW = nil--image.display(image.lena())
LOGGING_ACTIONS = false
IS_INCEPTION = false
IS_RESNET = false
DIFFERENT_FORMAT = IS_INCEPTION or IS_RESNET
MEAN_MODEL = torch.ones(3):double()*0.5
STD_MODEL = torch.ones(3):double()*0.5
MEAN_MODEL = torch.DoubleTensor({ 0.485, 0.456, 0.406 })
STD_MODEL = torch.DoubleTensor({ 0.229, 0.224, 0.225 })
IM_LENGTH = 200
IM_HEIGHT = 200

function addLeadingZero(number)
    -- Returns a string with a leading zero of the number if the number has only one digit (for model logging and sorting purposes)
    if number >= 0 and number <= 9 then
        return "0" .. number
    else
        return tostring(number)
    end
end

function set_hyperparams(params)
    --defaults: NOTE: APPARENTLY THE DEFAULTS PROVIDED TO THE COMMAND LINE
    -- DEFINITION IN THIS PROGRAM, ARE NOT TAKEN NITO ACCOUNT, WE SET THEM HERE
    if params.use_cuda then
        USE_CUDA = params.use_cuda
    -- else
    --     USE_CUDA = true
    end
    if params.use_continuous then
        USE_CONTINUOUS = params.use_continuous
    -- else
    --     USE_CONTINUOUS = true
    end
    if params.mcd then
        MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD = params.mcd
    -- else
    --     MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD = 0.4
    end
    if params.sigma then
        CONTINUOUS_ACTION_SIGMA = params.sigma
    -- else
    --     CONTINUOUS_ACTION_SIGMA = 0.6
    end
    if params.data_folder then
        DATA_FOLDER = params.data_folder
    -- else
    --     DATA_FOLDER = MOBILE_ROBOT --works best!
    end
    set_cuda_hyperparams(USE_CUDA)
    set_dataset_specific_hyperparams(DATA_FOLDER)
end

function set_cuda_hyperparams(USE_CUDA)
    --===========================================================
    -- CUDA CONSTANTS
    --===========================================================
    if USE_CUDA then
        require 'cunn'
        require 'cutorch'
        require 'cudnn'  --If trouble, installing, follow step 6 in https://github.com/jcjohnson/neural-style/blob/master/INSTALL.md
        tnt = require 'torchnet'
        vision = require 'torchnet-vision'  -- Install via https://github.com/Cadene/torchnet-vision
    end
    USE_SECOND_GPU = true
    if USE_CUDA and USE_SECOND_GPU then
      --  cutorch.setDevice(2)
    end
end

function set_dataset_specific_hyperparams(DATA_FOLDER)
    --======================================================
    --Continuous actions SETTINGS
    --======================================================
    --USE_CONTINUOUS = true --A switch between discrete and continuous actions (translates into calling getRandomBatchFromSeparateListContinuous instead of getRandomBatchFromSeparateList
    --ACTION_AMPLITUDE = 0.01
    -- The following parameter eliminates the need of finding close enough actions for assessing all priors except for the temporal.one.
    -- If the actions are too far away, they will make the gradient 0 and will not be considered for the update rule
    --CONTINUOUS_ACTION_SIGMA = 0.4
    --In contiuous actions, we take 2 actions, if they are very similar, the coef factor
    --is high (1 if the actions are the same), if not, the coef is close to 0. We add a constraint with the method
    --action_vectors_are_similar_enough to impose a cosine distance constraint when comparing actions, because the network will see a lot
    --of actions that are not similar, so instead of taking '2 random actions', we take '2 random actions, but above a certain similarity threshold'.
    --MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD = 0.6
    -- TODO shall it be different for each dataset depending on the variance of the input state space?
    --If so, What is a good proxy  parameter to set it?

    STRING_MEAN_AND_STD_FILE = PRELOAD_FOLDER..'meanStdImages_'..DATA_FOLDER..'.t7'
    now = os.date("*t")
    -- print('MODEL_ARCHITECTURE_FILE')
    -- print(MODEL_ARCHITECTURE_FILE) --./models/minimalNetModel
    -- print(MODEL_ARCHITECTURE_FILE:match("(.+)/(.+)")) -- returns  ./models	minimalNetModel
    _, architecture_name = MODEL_ARCHITECTURE_FILE:match("(.+)/(.+)") --architecture_name, _ = split(architecture_name, ".")

    if USE_CONTINUOUS then
        --DAY = 'Y'..now.year..'_D'..now.day..'_M'..now.month..'_H'..now.hour..'M'..now.min..'S'..now.sec..'_'..DATA_FOLDER..'_'..architecture_name..'_cont'..'_MCD0_'..(MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD*10)..'_S0_'..(CONTINUOUS_ACTION_SIGMA*10)
        DAY = 'Y'..now.year..'_D'..addLeadingZero(now.day)..'_M'..addLeadingZero(now.month)..'_H'..addLeadingZero(now.hour)..'M'..addLeadingZero(now.min)..'S'..addLeadingZero(now.sec)..'_'..DATA_FOLDER..'_'..architecture_name..'_cont'..'_MCD0_'..(MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD*10)..'_S0_'..(CONTINUOUS_ACTION_SIGMA*10)
    else
        --DAY = 'Y'..now.year..'_D'..now.day..'_M'..now.month..'_H'..now.hour..'M'..now.min..'S'..now.sec..'_'..DATA_FOLDER..'_'..architecture_name
        DAY = 'Y'..now.year..'_D'..addLeadingZero(now.day)..'_M'..addLeadingZero(now.month)..'_H'..addLeadingZero(now.hour)..'M'..addLeadingZero(now.min)..'S'..addLeadingZero(now.sec)..'_'..DATA_FOLDER..'_'..architecture_name
    end

    NAME_SAVE= 'model'..DAY
    SAVED_MODEL_PATH = LOG_FOLDER..NAME_SAVE

    if DATA_FOLDER == SIMPLEDATA3D then
       CLAMP_CAUSALITY = true  --TODO: make false when continuous works

       MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z doesn't really matter in fact
       MAX_TABLE = {0.8,0.7,10} -- for x,y,z doesn't really matter in fact

       DIMENSION_IN = 3

       REWARD_INDICE = 2
       INDEX_TABLE = {2,3,4} --column indice for coordinate in state file (respectively x,y,z)

       DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
       FILENAME_FOR_REWARD = "is_pressed"
       FILENAME_FOR_ACTION = "endpoint_action"
       FILENAME_FOR_STATE = "endpoint_state"

       SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
       AVG_FRAMES_PER_RECORD = 95

       MIN_TABLE = {0.42,-0.1,-10} -- for x,y,z doesn't really matter in fact
       MAX_TABLE = {0.75,0.6,10} -- for x,y,z doesn't really matter in fact

       DIMENSION_IN = 3
       DIMENSION_OUT = 3

       REWARD_INDEX = 2 --2 reward va_robot_limb_left_endpoint_state.txt"--endpoint_state"

       SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
       AVG_FRAMES_PER_RECORD = 100

    elseif DATA_FOLDER == MOBILE_ROBOT then
        print('Setting default hyperparams for MOBILE_ROBOT')
       --NOTE: DEFAULT PARAMETERS FOR OUR BASELINE DATABASE SET AT THE BEGINNING OF THE FILE (NEED TO BE DECLARED AS CONSTANTS)
    --    CLAMP_CAUSALITY = false--cant add to functions because it creates an import loop
       --
    --    MIN_TABLE = {-10000,-10000} -- for x,y
    --    MAX_TABLE = {10000,10000} -- for x,y
       --
    --    DIMENSION_IN = 2
    --    DIMENSION_OUT = 2  --worked just as well as 4 output dimensions
    --    REWARD_INDEX = 1  --3 reward values: -1, 0, 10
    --    INDEX_TABLE = {1,2} --column index for coordinate in state file (respectively x,y)
       --
    --    DEFAULT_PRECISION = 0.1
    --    FILENAME_FOR_ACTION = "recorded_robot_action.txt" --not used at all, we use state file, and compute the action with it (contains dx, dy)
    --    FILENAME_FOR_STATE = "recorded_robot_state.txt"
    --    FILENAME_FOR_REWARD = "recorded_robot_reward.txt"
       --
    --    -- WARNING : If you change the folder (top, pano, front)
    --    -- do rm preload_folder/* because the images won't be good
    --    SUB_DIR_IMAGE = 'recorded_camera_top'
    --    -- ====================================================
    --    AVG_FRAMES_PER_RECORD = 90

    elseif DATA_FOLDER == BABBLING then
      -- Leni's real Baxter data on  ISIR dataserver. It is named "data_archive_sim_1".
      --(real Baxter Pushing Objects).  If data is not converted into action, state
      -- and reward files with images in subfolder, run first the conversion tool from
      -- yml format to rgb based data in https://github.com/LeniLeGoff/DB_action_discretization
      --For the version 1 of the dataset, rewards are very sparse and not always there is 2 values for the reward (required to apply Causality prior)
      DEFAULT_PRECISION = 0.1
      CLAMP_CAUSALITY = false
      MIN_TABLE = {-10000, -10000, -10000} -- for x,y,z
      MAX_TABLE = {10000, 10000, 10000} -- for x,y, z
      --
      DIMENSION_IN = 3
      DIMENSION_OUT = 3
      REWARD_INDEX = 2 -- column (2 reward values: 0, 1 in this dataset)
      INDEX_TABLE = {2,3,4} --column indexes for coordinate in state file (respectively x,y)
      --
      FILENAME_FOR_REWARD = "reward_pushing_object.txt"  -- 1 if the object being pushed actually moved
      FILENAME_FOR_STATE = "state_pushing_object.txt" --computed while training based on action
      FILENAME_FOR_ACTION_DELTAS = "state_pushing_object_deltas.txt"
      FILENAME_FOR_ACTION = FILENAME_FOR_ACTION_DELTAS --""action_pushing_object.txt"

      SUB_DIR_IMAGE = 'baxter_pushing_objects'
      AVG_FRAMES_PER_RECORD = 60
      -- Causality needs at least 2 different values of reward and in sparse dataset such as babbling_1, this does not occur always
      --PRIORS_TO_APPLY ={{"Rep","Prop","Temp"}}
      PRIORS_CONFIGS_TO_APPLY ={{"Temp"}}--, works best than 3 priors for babbling so far  {"Prop","Temp"}, {"Prop","Rep"},  {"Temp","Rep"}, {"Prop","Temp","Rep"}}  --TODO report 1 vs 2 vs 3 priors, add all prioris when Babbling contains +1 reward value
      --print('WARNING: Causality prior, at least, will be ignored for dataset because of too sparse rewards (<2 value types). TODO: convert to 3 reward values'..BABBLING?

    elseif DATA_FOLDER == PUSHING_BUTTON_AUGMENTED then
        CLAMP_CAUSALITY = true --TODO: make false when continuous works

        MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z
        MAX_TABLE = {0.8,0.7,10} -- for x,y,z

        DIMENSION_IN = 3
        DIMENSION_OUT = 3

        REWARD_INDEX = 2 --2 reward values: -0, 1 ?
        INDEX_TABLE = {2,3,4} --column index for coordinates in state file, respectively (x,y,z)

        DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
        FILENAME_FOR_REWARD = "recorded_button1_is_pressed.txt"--"is_pressed"
        FILENAME_FOR_ACTION = "recorded_robot_limb_left_endpoint_action.txt"--endpoint_action"  -- Never written, always computed on the fly
        FILENAME_FOR_STATE = "recorded_robot_limb_left_endpoint_state.txt"--endpoint_state"

        SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
        AVG_FRAMES_PER_RECORD = 100

    elseif DATA_FOLDER == STATIC_BUTTON_SIMPLEST then  -- TODO if nothing changes, add OR to previous case
        CLAMP_CAUSALITY = true --TODO: make false when continuous works

        MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z
        MAX_TABLE = {0.8,0.7,10} -- for x,y,z

        DIMENSION_IN = 3
        DIMENSION_OUT = 3

        REWARD_INDEX = 2 --2 reward values: -0, 1 ?
        INDEX_TABLE = {2,3,4} --column index for coordinates in state file, respectively (x,y,z)

        DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
        FILENAME_FOR_REWARD = "recorded_button1_is_pressed.txt"
        FILENAME_FOR_ACTION = "recorded_robot_limb_left_endpoint_action.txt" -- Never written, always computed on the fly
        FILENAME_FOR_STATE = "recorded_robot_limb_left_endpoint_state.txt"

        SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
        AVG_FRAMES_PER_RECORD = 90

    else
      print("No supported data folder provided, please add either of the data folders defined in hyperparams: "..BABBLING..", "..MOBILE_ROBOT.." "..SIMPLEDATA3D..' or others in const.lua' )
      os.exit()
    end


    if VISUALIZE_IMAGES_TAKEN or VISUALIZE_CAUS_IMAGE or VISUALIZE_IMAGE_CROP or VISUALIZE_MEAN_STD or VISUALIZE_AE then
       --Creepy, but need a placeholder, to prevent many window to pop
       WINDOW = image.display(image.lena())
    end

    LOGGING_ACTIONS = false

    --IS_INCEPTION = string.find(MODEL_ARCHITECTURE_FILE, 'inception')
    -- since the model require images to be a 3x299x299, and normalize differently, we need to adapt
    --IS_RESNET = string.find(MODEL_ARCHITECTURE_FILE, 'resnet')
    if string.find(MODEL_ARCHITECTURE_FILE, 'inception') then
        IS_INCEPTION = true
    end
    if string.find(MODEL_ARCHITECTURE_FILE, 'resnet') then
        -- since the model require images to be a 3x299x299, and normalize differently, we need to adapt
        IS_RESNET = true
    end

    DIFFERENT_FORMAT = IS_INCEPTION or IS_RESNET

    if IS_INCEPTION then
       IM_LENGTH = 299
       IM_HEIGHT = 299
       MEAN_MODEL = torch.ones(3):double()*0.5
       STD_MODEL = torch.ones(3):double()*0.5
    elseif IS_RESNET then
       IM_LENGTH = 224
       IM_HEIGHT = 224
       MEAN_MODEL = torch.DoubleTensor({ 0.485, 0.456, 0.406 })
       STD_MODEL = torch.DoubleTensor({ 0.229, 0.224, 0.225 })
    else
       IM_LENGTH = 200
       IM_HEIGHT = 200
    end

    if params then
        print_hyperparameters()
    end
end

function print_hyperparameters()
    print("Model :",MODEL_ARCHITECTURE_FILE)
    print("\nUSE_CUDA ",USE_CUDA," \nUSE_CONTINUOUS ACTIONS: ",USE_CONTINUOUS,'\nMAX_COS_DIST_AMONG_ACTIONS_THRESHOLD: ',MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD,' CONTINUOUS_ACTION_SIGMA: ', CONTINUOUS_ACTION_SIGMA)
    print("============ DATA USED =========\n",
                    DATA_FOLDER,
      "\n================================")
end




-- if DATA_FOLDER == SIMPLEDATA3D then
--    CLAMP_CAUSALITY = true  --TODO: make false when continuous works
--
--    MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z doesn't really matter in fact
--    MAX_TABLE = {0.8,0.7,10} -- for x,y,z doesn't really matter in fact
--
--    DIMENSION_IN = 3
--
--    REWARD_INDICE = 2
--    INDICE_TABLE = {2,3,4} --column indice for coordinate in state file (respectively x,y,z)
--
--    DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
--    FILENAME_FOR_REWARD = "is_pressed"
--    FILENAME_FOR_ACTION = "endpoint_action"
--    FILENAME_FOR_STATE = "endpoint_state"
--
--    SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
--    AVG_FRAMES_PER_RECORD = 95
--
--    MIN_TABLE = {0.42,-0.1,-10} -- for x,y,z doesn't really matter in fact
--    MAX_TABLE = {0.75,0.6,10} -- for x,y,z doesn't really matter in fact
--
--    DIMENSION_IN = 3
--    DIMENSION_OUT = 3
--
--    REWARD_INDEX = 2 --2 reward va_robot_limb_left_endpoint_state.txt"--endpoint_state"
--
--    SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
--    AVG_FRAMES_PER_RECORD = 100
--
-- elseif DATA_FOLDER == MOBILE_ROBOT then
--
--    CLAMP_CAUSALITY = false--cant add to functions because it creates an import loop
--
--    MIN_TABLE = {-10000,-10000} -- for x,y
--    MAX_TABLE = {10000,10000} -- for x,y
--
--    DIMENSION_IN = 2
--    DIMENSION_OUT = 2  --worked just as well as 4 output dimensions
--    REWARD_INDEX = 1  --3 reward values: -1, 0, 10
--    INDEX_TABLE = {1,2} --column index for coordinate in state file (respectively x,y)
--
--    DEFAULT_PRECISION = 0.1
--    FILENAME_FOR_ACTION = "recorded_robot_action.txt" --not used at all, we use state file, and compute the action with it (contains dx, dy)
--    FILENAME_FOR_STATE = "recorded_robot_state.txt"
--    FILENAME_FOR_REWARD = "recorded_robot_reward.txt"
--
--    -- WARNING : If you change the folder (top, pano, front)
--    -- do rm preload_folder/* because the images won't be good
--    SUB_DIR_IMAGE = 'recorded_camera_top'
--    -- ====================================================
--
--    AVG_FRAMES_PER_RECORD = 90
--
-- elseif DATA_FOLDER == BABBLING then
--   -- Leni's real Baxter data on  ISIR dataserver. It is named "data_archive_sim_1".
--   --(real Baxter Pushing Objects).  If data is not converted into action, state
--   -- and reward files with images in subfolder, run first the conversion tool from
--   -- yml format to rgb based data in https://github.com/LeniLeGoff/DB_action_discretization
--   --For the version 1 of the dataset, rewards are very sparse and not always there is 2 values for the reward (required to apply Causality prior)
--   DEFAULT_PRECISION = 0.1
--   CLAMP_CAUSALITY = false
--   MIN_TABLE = {-10000, -10000, -10000} -- for x,y,z
--   MAX_TABLE = {10000, 10000, 10000} -- for x,y, z
--   --
--   DIMENSION_IN = 3
--   DIMENSION_OUT = 3
--   REWARD_INDEX = 2 -- column (2 reward values: 0, 1 in this dataset)
--   INDEX_TABLE = {2,3,4} --column indexes for coordinate in state file (respectively x,y)
--   --
--   FILENAME_FOR_REWARD = "reward_pushing_object.txt"  -- 1 if the object being pushed actually moved
--   FILENAME_FOR_STATE = "state_pushing_object.txt" --computed while training based on action
--   FILENAME_FOR_ACTION_DELTAS = "state_pushing_object_deltas.txt"
--   FILENAME_FOR_ACTION = FILENAME_FOR_ACTION_DELTAS --""action_pushing_object.txt"
--
--   SUB_DIR_IMAGE = 'baxter_pushing_objects'
--   AVG_FRAMES_PER_RECORD = 60
--   -- Causality needs at least 2 different values of reward and in sparse dataset such as babbling_1, this does not occur always
--   --PRIORS_TO_APPLY ={{"Rep","Prop","Temp"}}
--   PRIORS_CONFIGS_TO_APPLY ={{"Temp"}}--, works best than 3 priors for babbling so far  {"Prop","Temp"}, {"Prop","Rep"},  {"Temp","Rep"}, {"Prop","Temp","Rep"}}  --TODO report 1 vs 2 vs 3 priors, add all prioris when Babbling contains +1 reward value
--   --print('WARNING: Causality prior, at least, will be ignored for dataset because of too sparse rewards (<2 value types). TODO: convert to 3 reward values'..BABBLING?
--
-- elseif DATA_FOLDER == PUSHING_BUTTON_AUGMENTED then
--     CLAMP_CAUSALITY = true --TODO: make false when continuous works
--
--     MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z
--     MAX_TABLE = {0.8,0.7,10} -- for x,y,z
--
--     DIMENSION_IN = 3
--     DIMENSION_OUT = 3
--
--     REWARD_INDEX = 2 --2 reward values: -0, 1 ?
--     INDEX_TABLE = {2,3,4} --column index for coordinates in state file, respectively (x,y,z)
--
--     DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
--     FILENAME_FOR_REWARD = "recorded_button1_is_pressed.txt"--"is_pressed"
--     FILENAME_FOR_ACTION = "recorded_robot_limb_left_endpoint_action.txt"--endpoint_action"  -- Never written, always computed on the fly
--     FILENAME_FOR_STATE = "recorded_robot_limb_left_endpoint_state.txt"--endpoint_state"
--
--     SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
--     AVG_FRAMES_PER_RECORD = 100
--
-- elseif DATA_FOLDER == STATIC_BUTTON_SIMPLEST then  -- TODO if nothing changes, add OR to previous case
--     CLAMP_CAUSALITY = true --TODO: make false when continuous works
--
--     MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z
--     MAX_TABLE = {0.8,0.7,10} -- for x,y,z
--
--     DIMENSION_IN = 3
--     DIMENSION_OUT = 3
--
--     REWARD_INDEX = 2 --2 reward values: -0, 1 ?
--     INDEX_TABLE = {2,3,4} --column index for coordinates in state file, respectively (x,y,z)
--
--     DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
--     FILENAME_FOR_REWARD = "recorded_button1_is_pressed.txt"--"is_pressed"
--     FILENAME_FOR_ACTION = "recorded_robot_limb_left_endpoint_action.txt"--endpoint_action"  -- Never written, always computed on the fly
--     FILENAME_FOR_STATE = "recorded_robot_limb_left_endpoint_state.txt"--endpoint_state"
--
--     SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
--     AVG_FRAMES_PER_RECORD = 90
--
-- else
--   print("No supported data folder provided, please add either of the data folders defined in hyperparams: "..BABBLING..", "..MOBILE_ROBOT.." "..SIMPLEDATA3D..' or others in const.lua' )
--   os.exit()
-- end
