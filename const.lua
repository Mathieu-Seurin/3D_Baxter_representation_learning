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
require 'cutorch'
require 'hyperparams'
--torch.manualSeed(100)


--===========================================================
-- CUDA CONSTANTS
--===========================================================
USE_CUDA = false
USE_SECOND_GPU = true

if USE_CUDA and USE_SECOND_GPU then
   cutorch.setDevice(2)
end

--=====================================
--DATA AND LOG FOLDER NAME etc..
--====================================
PRELOAD_FOLDER = 'preload_folder/'
lfs.mkdir(PRELOAD_FOLDER)

LOG_FOLDER = 'Log/'
MODEL_PATH = LOG_FOLDER

MODEL_ARCHITECTURE_FILE = './models/topUniqueSimplerWOTanh'

STRING_MEAN_AND_STD_FILE = PRELOAD_FOLDER..'meanStdImages_'..DATA_FOLDER..'.t7'
LEARNED_REPRESENTATIONS_FILE = "saveImagesAndRepr.txt"

now = os.date("*t")
if USE_CONTINUOUS then
    DAY = now.year..'_'..now.yday..'_'..now.hour..'_'..now.min..'_'..now.sec..'_'..DATA_FOLDER..'_cont'
else
    DAY = now.year..'_'..now.yday..'_'..now.hour..'_'..now.min..'_'..now.sec..'_'..DATA_FOLDER
end
NAME_SAVE= 'model'..DAY
RELOAD_MODEL = false

--===========================================================
-- VISUALIZATION TOOL
-- if you want to visualize images, use 'qlua' instead of 'th'
--===========================================================
VISUALIZE_IMAGES_TAKEN = false
VISUALIZE_CAUS_IMAGE = false
VISUALIZE_IMAGE_CROP = false
VISUALIZE_MEAN_STD = false

if VISUALIZE_IMAGES_TAKEN or VISUALIZE_CAUS_IMAGE or VISUALIZE_IMAGE_CROP or VISUALIZE_MEAN_STD then
   --Creepy, but need a placeholder, to prevent many window to pop
   WINDOW = image.display(image.lena())
end

IM_LENGTH = 200
IM_HEIGHT = 200
IM_CHANNEL = 3 --image channels (RGB)

--================================================
-- dataFolder specific constants : filename, dim_in, indexes in state file etc...
--===============================================
if DATA_FOLDER == SIMPLEDATA3D then
   CLAMP_CAUSALITY = true

   MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z
   MAX_TABLE = {0.8,0.7,10} -- for x,y,z

   DIMENSION_IN = 3
   DIMENSION_OUT = 3 --TODO better specify here than leave it up to the model?

   REWARD_INDEX = 2 --2 reward values: -0, 1
   INDEX_TABLE = {2,3,4} --column index for coordinates in state file, respectively (x,y,z)

   DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
   FILENAME_FOR_REWARD = "recorded_button1_is_pressed.txt"--"is_pressed"
   FILENAME_FOR_ACTION = "recorded_robot_limb_left_endpoint_action.txt"--endpoint_action"
   FILENAME_FOR_STATE = "recorded_robot_limb_left_endpoint_state.txt"--endpoint_state"

   SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
   AVG_FRAMES_PER_RECORD = 1000

elseif DATA_FOLDER == MOBILE_ROBOT then

   CLAMP_CAUSALITY = false

   MIN_TABLE = {-10000,-10000} -- for x,y
   MAX_TABLE = {10000,10000} -- for x,y

   DIMENSION_IN = 2
   DIMENSION_OUT = 2
   REWARD_INDEX = 1  --3 reward values: -1, 0, 10
   INDEX_TABLE = {1,2} --column index for coordinate in state file (respectively x,y)

   DEFAULT_PRECISION = 0.1
   FILENAME_FOR_ACTION = "recorded_robot_action.txt" --not used at all, we use state file, and compute the action with it (contains dx, dy)
   FILENAME_FOR_STATE = "recorded_robot_state.txt"
   FILENAME_FOR_REWARD = "recorded_robot_reward.txt"

   SUB_DIR_IMAGE = 'recorded_camera_top'
   AVG_FRAMES_PER_RECORD = 90


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

else
  print("No supported data folder provided, please add either of the data folders defined in hyperparams: "..BABBLING..", "..MOBILE_ROBOT.." "..SIMPLEDATA3D )
  os.exit()
end


FILE_PATTERN_TO_EXCLUDE = 'deltas'
print("\nUSE_CUDA ",USE_CUDA," \nUSE_CONTINUOUS ACTIONS: ",USE_CONTINUOUS)
