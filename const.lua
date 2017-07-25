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

---NOTE: THESE ARE DEFAULTS (IF NOT COMMAND LINE ARGS ARE PASSED), AND ARE OVERRIDEN BY DATA_FOLDER SPECIFIC CASES BELOW :
----------------------------------------------------------------------------------------------------------------------------
USE_CUDA = true
USE_SECOND_GPU = false
USE_CONTINUOUS = true
MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD = 0.5
CONTINUOUS_ACTION_SIGMA = 0.5
DATA_FOLDER = MOBILE_ROBOT --works best!

if USE_CUDA then
    require 'cunn'
    require 'cutorch'
    require 'cudnn'  --If trouble, installing, follow step 6 in https://github.com/jcjohnson/neural-style/blob/master/INSTALL.md
    -- and https://github.com/soumith/cudnn.torch  --TODO: set to true when speed issues rise
    -- cudnn.benchmark = true -- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
    --                -- If this is set to false, uses some in-built heuristics that might not always be fastest.
    -- cudnn.fastest = true -- this is like the :fastest() mode for the Convolution modules,
                 -- simply picks the fastest convolution algorithm, rather than tuning for workspace size
    tnt = require 'torchnet'
    vision = require 'torchnet-vision'  -- Install via https://github.com/Cadene/torchnet-vision
end

if USE_CUDA and USE_SECOND_GPU then
   cutorch.setDevice(2)
end

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

RELOAD_MODEL = false

--===========================================================
-- VISUALIZATION TOOL
-- if you want to visualize images, use 'qlua' instead of 'th'
--===========================================================
VISUALIZE_IMAGES_TAKEN = false -- true for visualizing images taken in each prior
VISUALIZE_CAUS_IMAGE = false
VISUALIZE_IMAGE_CROP = false
VISUALIZE_MEAN_STD = false
VISUALIZE_AE = false

IM_CHANNEL = 3 --image channels (RGB)
ACTION_AMPLITUDE = 0.01
--================================================
-- dataFolder specific constants : filename, dim_in, indexes in state file etc...
--================================================
CLAMP_CAUSALITY = false

MIN_TABLE = {-10000,-10000} -- for x,y
MAX_TABLE = {10000,10000} -- for x,y

DIMENSION_IN = 2

REWARD_INDEX = 1  --3 reward values: -1, 0, 10
INDEX_TABLE = {1,2} --column index for coordinate in state file (respectively x,y)

DEFAULT_PRECISION = 0.05
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
CAN_HOLD_ALL_SEQ_IN_RAM = true
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
    if number >= 0 and number <= 9 then  return "0" .. number else return tostring(number)    end
end

---------------------------------------------------------------------------------------
-- Function :	set_hyperparams(params)
--======================================================
--Continuous actions SETTINGS
--======================================================
--USE_CONTINUOUS = true --A switch between discrete and continuous actions (translates into calling getRandomBatchFromSeparateListContinuous instead of getRandomBatchFromSeparateList
--ACTION_AMPLITUDE = 0.01
-- The following parameter eliminates the need of finding close enough actions for assessing all priors except for the temporal.one.
-- If the actions are too far away, they will make the gradient 0 and will not be considered for the update rule
--CONTINUOUS_ACTION_SIGMA = 0.4
    -- TODO shall it be different (and saevd into CONFIG.TXT file (TODO) for each dataset depending on the variance of the input state space?
--If so, What is a good proxy  parameter to set it?
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function set_hyperparams(params)
    --overriding the defaults:
    USE_CUDA = params.use_cuda      --print ('Boolean param: ') -- type is boolean
    USE_CONTINUOUS = params.use_continuous
    MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD = params.mcd
    CONTINUOUS_ACTION_SIGMA = params.sigma
    DATA_FOLDER = params.data_folder  --print('[Log: Setting command line dataset to '..params.data_folder..']') type is a str
    set_cuda_hyperparams(USE_CUDA)
    set_dataset_specific_hyperparams(DATA_FOLDER)
end

function set_cuda_hyperparams(USE_CUDA)
    --===========================================================
    -- CUDA CONSTANTS
    --===========================================================
    -- if USE_CUDA then
    --     require 'cunn'
    --     require 'cutorch'
    --     require 'cudnn'  --If trouble, installing, follow step 6 in https://github.com/jcjohnson/neural-style/blob/master/INSTALL.md
    --     -- and https://github.com/soumith/cudnn.torch
    --     cudnn.benchmark = true -- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
    --                    -- If this is set to false, uses some in-built heuristics that might not always be fastest.
    --     cudnn.fastest = true -- this is like the :fastest() mode for the Convolution modules,
    --                  -- simply picks the fastest convolution algorithm, rather than tuning for workspace size
    --     tnt = require 'torchnet'
    --     vision = require 'torchnet-vision'  -- Install via https://github.com/Cadene/torchnet-vision
    -- end
    if USE_CUDA and USE_SECOND_GPU then
       cutorch.setDevice(2)
    end
end

function set_dataset_specific_hyperparams(DATA_FOLDER)
    STRING_MEAN_AND_STD_FILE = PRELOAD_FOLDER..'meanStdImages_'..DATA_FOLDER..'.t7'
    now = os.date("*t")
    -- print(MODEL_ARCHITECTURE_FILE) --./models/minimalNetModel      -- print(MODEL_ARCHITECTURE_FILE:match("(.+)/(.+)")) -- returns  ./models	minimalNetModel
    _, architecture_name = MODEL_ARCHITECTURE_FILE:match("(.+)/(.+)") --architecture_name, _ = split(architecture_name, ".")

    if USE_CONTINUOUS then
        DAY = 'Y'..now.year..'_D'..addLeadingZero(now.day)..'_M'..addLeadingZero(now.month)..'_H'..addLeadingZero(now.hour)..'M'..addLeadingZero(now.min)..'S'..addLeadingZero(now.sec)..'_'..DATA_FOLDER..'_'..architecture_name..'_cont'..'_MCD'..MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD..'_S'..CONTINUOUS_ACTION_SIGMA
        DAY = DAY:gsub("%.", "_")  -- replace decimal points by '_' for folder naming
    else
        DAY = 'Y'..now.year..'_D'..addLeadingZero(now.day)..'_M'..addLeadingZero(now.month)..'_H'..addLeadingZero(now.hour)..'M'..addLeadingZero(now.min)..'S'..addLeadingZero(now.sec)..'_'..DATA_FOLDER..'_'..architecture_name
    end

    NAME_SAVE= 'model'..DAY
    SAVED_MODEL_PATH = LOG_FOLDER..NAME_SAVE

    if DATA_FOLDER == SIMPLEDATA3D then
       DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
       CLAMP_CAUSALITY = true  --TODO: make false when continuous works

       MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z doesn't really matter in fact
       MAX_TABLE = {0.8,0.7,10} -- for x,y,z doesn't really matter in fact

       DIMENSION_IN = 3

       REWARD_INDICE = 2
       INDEX_TABLE = {2,3,4} --column indice for coordinate in state file (respectively x,y,z)

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
        print('Setting hyperparams for MOBILE_ROBOT (our baseline reproducing Jonchowscki)')
       --NOTE: DEFAULT PARAMETERS FOR OUR BASELINE DATABASE SET AT THE BEGINNING OF THE FILE (NEED TO BE DECLARED AS CONSTANTS
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

    elseif DATA_FOLDER == STATIC_BUTTON_SIMPLEST then
        CLAMP_CAUSALITY = false --TODO: make false when continuous works
        -- A point where the robot wants the state to be very similar. Like a reference point for the robot

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
        AVG_FRAMES_PER_RECORD = 90  --HINT: reduce for fast full epoch testing in CPU mode

        if BRING_CLOSER_REWARD then
           PRIORS_CONFIGS_TO_APPLY ={{"Temp","Rep","Prop","Caus","make_reward_closer"}}
        end

        if BRING_CLOSER_REF_POINT then
           PRIORS_CONFIGS_TO_APPLY ={{"Temp","Rep","Prop","Caus","fixed_point"}}

           FIXED_POS = {0.587, -0.025, -0.143}
           ROUNDING_VALUE_FIX = 0.02

           FIXED_POS = {0.673, -0.0888, 0.087}
           ROUNDING_VALUE_FIX = 0.02

           FIXED_POS = {0.633, -0.034, 0.138}
           ROUNDING_VALUE_FIX = 0.02

           -- is the position the same as the reference point, for this rounding ?
           -- See arrondit for more details
        end

    elseif DATA_FOLDER == COMPLEX_DATA then
        CLAMP_CAUSALITY = false --TODO: make false when continuous works

        FIXED_POS = {0.587, -0.036, -0.143}
        -- A point where the robot wants the state to be very similar. Like a reference point for the robot

        MIN_TABLE = {0.42,-0.1,-0.11} -- for x,y,z
        MAX_TABLE = {0.75,0.60,0.35} -- for x,y,z

        DIMENSION_IN = 3
        DIMENSION_OUT = 3

        REWARD_INDEX = 2 -- Which column in the reward file indicates the value of reward ?
        INDEX_TABLE = {2,3,4} --column index for coordinates in state file, respectively (x,y,z)

        DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
        FILENAME_FOR_REWARD = "recorded_button1_is_pressed.txt"
        FILENAME_FOR_ACTION = "recorded_robot_limb_left_endpoint_action.txt" -- Never written, always computed on the fly
        FILENAME_FOR_STATE = "recorded_robot_limb_left_endpoint_state.txt"

        SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
        AVG_FRAMES_PER_RECORD = 200  --HINT: reduce for fast full epoch testing in CPU mode

        if BRING_CLOSER_REWARD then
           PRIORS_CONFIGS_TO_APPLY ={{"Temp","Rep","Prop","Caus","make_reward_closer"}}
        end

        if BRING_CLOSER_REF_POINT then
           PRIORS_CONFIGS_TO_APPLY ={{"Temp","Rep","Prop","Caus","fixed_pos"}}
        end
    elseif DATA_FOLDER == COLORFUL then
        CLAMP_CAUSALITY = false --TODO: make false when continuous works

        FIXED_POS  = {0.6, 0.30, 0.10} -- starting point for every sequence
        -- FIXED_POS = {0.587, -0.036, -0.143}
        -- A point where the robot wants the state to be very similar. Like a reference point for the robot

        MIN_TABLE = {0.42,-0.1,-0.11} -- for x,y,z
        MAX_TABLE = {0.75,0.60,0.35} -- for x,y,z

        DIMENSION_IN = 3
        DIMENSION_OUT = 3

        REWARD_INDEX = 2 -- Which column in the reward file indicates the value of reward ?
        INDEX_TABLE = {2,3,4} --column index for coordinates in state file, respectively (x,y,z)

        DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
        FILENAME_FOR_REWARD = "recorded_button1_is_pressed.txt"
        FILENAME_FOR_ACTION = "recorded_robot_limb_left_endpoint_action.txt" -- Never written, always computed on the fly
        FILENAME_FOR_STATE = "recorded_robot_limb_left_endpoint_state.txt"

        SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
        AVG_FRAMES_PER_RECORD = 250  --HINT: reduce for fast full epoch testing in CPU mode

        if BRING_CLOSER_REWARD then
           PRIORS_CONFIGS_TO_APPLY ={{"Temp","Rep","Prop","Caus","make_reward_closer"}}
        end

        if BRING_CLOSER_REF_POINT then
           PRIORS_CONFIGS_TO_APPLY ={{"Temp","Rep","Prop","Caus","fixed_pos"}}
        end

        if USE_CONTINUOUS then
            DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
        end
    else
      print("No supported data folder provided, please add either of the data folders defined in hyperparams: "..BABBLING..", "..MOBILE_ROBOT.." "..SIMPLEDATA3D..' or others in const.lua' )
      os.exit()
    end

    MIN_TABLE = {0.42,-0.09,-10} -- for x,y,z
    MAX_TABLE = {0.74,0.59,10} -- for x,y,z

    if VISUALIZE_IMAGES_TAKEN or VISUALIZE_CAUS_IMAGE or VISUALIZE_IMAGE_CROP or VISUALIZE_MEAN_STD or VISUALIZE_AE then
       --Creepy, but need a placeholder, to prevent many window to pop
       WINDOW = image.display(image.lena())
    end

    LOGGING_ACTIONS = false

    if string.find(MODEL_ARCHITECTURE_FILE, 'inception') then
        IS_INCEPTION = true
    end
    if string.find(MODEL_ARCHITECTURE_FILE, 'resnet') then
        -- since the model require images to be a 3x299x299, and normalize differently, we need to adapt
        IS_RESNET = true
    end
    if string.find(MODEL_ARCHITECTURE_FILE, 'minimalNetModel') then --TODO replace by constants
        error([[minimalNetModel should only be used in learn_autoencoder.lua, not script.lua]])
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

    print_hyperparameters()
end

function print_hyperparameters(extra_string_to_print)
    print(extra_string_to_print)
    print("============ Experiment: DATA_FOLDER USED =========\n",
                        DATA_FOLDER,
												" (LOG_FOLDER ", LOG_FOLDER,
                        ")\nUSE_CUDA ",USE_CUDA,", USE_CONTINUOUS ACTIONS: ",USE_CONTINUOUS, " MODEL: ",MODEL_ARCHITECTURE_FILE,". PRIORS_CONFIGS_TO_APPLY", PRIORS_CONFIGS_TO_APPLY)
    print('EXTRAPOLATE_ACTION ','EXTRAPOLATE_ACTION_CAUS ','BRING_CLOSER_REWARD ','BRING_CLOSER_REF_POINT: ')
    print(EXTRAPOLATE_ACTION,EXTRAPOLATE_ACTION_CAUS,BRING_CLOSER_REWARD,BRING_CLOSER_REF_POINT)
    if USE_CONTINUOUS then  --otherwise, it is not used
        print ('MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD: ',MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD,' CONTINUOUS_ACTION_SIGMA: ', CONTINUOUS_ACTION_SIGMA)
    end
    print("\n================================")
end

-- 49 (1 repeated by error) IMAGES TEST SET HANDPICKED TO SHOW VISUAL VARIABILITY
IMG_TEST_SET = {
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00042.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00065.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00048.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00080.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00004.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00078.jpg',

'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00056.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00047.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00033.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00005.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00026.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00056.jpg',

'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00056.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00063.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00035.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',

'staticButtonSimplest/record_019/recorded_cameras_head_camera_2_image_compressed/frame00009.jpg',
'staticButtonSimplest/record_019/recorded_cameras_head_camera_2_image_compressed/frame00074.jpg',
'staticButtonSimplest/record_019/recorded_cameras_head_camera_2_image_compressed/frame00049.jpg',

'staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed/frame00085.jpg',
'staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',

'staticButtonSimplest/record_031/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'staticButtonSimplest/record_031/recorded_cameras_head_camera_2_image_compressed/frame00007.jpg',
'staticButtonSimplest/record_031/recorded_cameras_head_camera_2_image_compressed/frame00070.jpg',

'staticButtonSimplest/record_036/recorded_cameras_head_camera_2_image_compressed/frame00085.jpg',
'staticButtonSimplest/record_036/recorded_cameras_head_camera_2_image_compressed/frame00023.jpg',
'staticButtonSimplest/record_036/recorded_cameras_head_camera_2_image_compressed/frame00036.jpg',

'staticButtonSimplest/record_037/recorded_cameras_head_camera_2_image_compressed/frame00053.jpg',
'staticButtonSimplest/record_037/recorded_cameras_head_camera_2_image_compressed/frame00083.jpg',
'staticButtonSimplest/record_037/recorded_cameras_head_camera_2_image_compressed/frame00032.jpg',

'staticButtonSimplest/record_040/recorded_cameras_head_camera_2_image_compressed/frame00045.jpg',
'staticButtonSimplest/record_040/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'staticButtonSimplest/record_040/recorded_cameras_head_camera_2_image_compressed/frame00080.jpg',

'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00034.jpg',
'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00059.jpg',
'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00089.jpg',
'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00030.jpg',

'staticButtonSimplest/record_050/recorded_cameras_head_camera_2_image_compressed/frame00064.jpg',
'staticButtonSimplest/record_050/recorded_cameras_head_camera_2_image_compressed/frame00019.jpg',
'staticButtonSimplest/record_050/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',

'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',
'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00068.jpg',
'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00025.jpg'}
