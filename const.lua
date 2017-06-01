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
--torch.manualSeed(100)

--=====================================
--DATA AND LOG FOLDER NAME etc..
--====================================
PRELOAD_FOLDER = 'preload_folder/'
lfs.mkdir(PRELOAD_FOLDER)

LOG_FOLDER = 'Log/'
MODEL_PATH = LOG_FOLDER

MODEL_ARCHITECTURE_FILE = './models/topUniqueSimplerWOTanh'

STRING_MEAN_AND_STD_FILE = PRELOAD_FOLDER..'meanStdImages_'..DATA_FOLDER..'.t7'

now = os.date("*t")
DAY = now.year..'_'..now.yday..'__'..now.hour..'_'..now.min..'_'..now.sec
NAME_SAVE= 'model'..DAY
RELOAD_MODEL = false

CAN_HOLD_ALL_SEQ_IN_RAM = true
-- indicates of you can hold all images sequences in your RAM or not, that way, you can compute much faster.

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

IM_CHANNEL = 3
IM_LENGTH = 200
IM_HEIGHT = 200

--===========================================================
-- CUDA CONSTANTS
--===========================================================
USE_CUDA = true
USE_SECOND_GPU = true

if USE_CUDA and USE_SECOND_GPU then
   require 'cutorch'
   cutorch.setDevice(2)
end

--================================================
-- dataFolder specific constants : filename, dim_in, indices in state file etc...
--===============================================

if DATA_FOLDER == 'simpleData3D' then
   CLAMP_CAUSALITY = true

   MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z
   MAX_TABLE = {0.8,0.7,10} -- for x,y,z

   DIMENSION_IN = 3

   REWARD_INDICE = 2
   INDICE_TABLE = {2,3,4} --column indice for coordinate in state file (respectively x,y,z)

   DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
   FILENAME_FOR_REWARD = "is_pressed"
   FILENAME_FOR_ACTION = "endpoint_action"
   FILENAME_FOR_STATE = "endpoint_state"

   SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
   AVG_FRAMES_PER_RECORD = 1000

elseif DATA_FOLDER == 'mobileRobot' then

   CLAMP_CAUSALITY = false

   MIN_TABLE = {-10000,-10000} -- for x,y
   MAX_TABLE = {10000,10000} -- for x,y

   DIMENSION_IN = 2

   REWARD_INDICE = 1
   INDICE_TABLE = {1,2} --column indice for coordinate in state file (respectively x,y)

   DEFAULT_PRECISION = 0.1
   FILENAME_FOR_ACTION = "action" --not used at all, we use state file, and compute the action with it (contains dx, dy)
   FILENAME_FOR_STATE = "state"
   FILENAME_FOR_REWARD = "reward"

   SUB_DIR_IMAGE = 'recorded_camera_top'
   AVG_FRAMES_PER_RECORD = 90


elseif DATA_FOLDER == 'babbling_converted' then
  -- Leni's real Baxter data on  ISIR dataserver. It is named "data_archive_sim_1".
  --(real Baxter Pushing Objects).  If data is not converted into action, state
  -- and reward files with images in subfolder, run first the conversion tool from
  -- yml format to rgb based data in https://github.com/LeniLeGoff/DB_action_discretization
  DEFAULT_PRECISION = 0.1
  CLAMP_CAUSALITY = false
  MIN_TABLE = {-10000,-10000} -- for x,y
  MAX_TABLE = {10000,10000} -- for x,y
  --
  DIMENSION_IN = 3
  REWARD_INDICE = 2
  INDICE_TABLE = {2,3,4} --column indexes for coordinate in state file (respectively x,y)
  --
  FILENAME_FOR_REWARD = "reward_pushing_object.txt"  -- 1 if the object being pushed actually moved
  FILENAME_FOR_STATE = "state_pushing_object.txt" --computed while training based on action
  FILENAME_FOR_ACTION = "action_pushing_object.txt"
  --FILENAME_FOR_ACTION_DELTAS = "action_pushing_object_deltas.txt"

  SUB_DIR_IMAGE = 'baxter_pushing_objects'
  AVG_FRAMES_PER_RECORD = 60

else
  print("No supported data folder provided, please add either of simpleData3D, mobileRobot or Leni's babbling")
  os.exit()
end


FILE_PATTERN_TO_EXCLUDE = 'deltas'
print("\nUSE_CUDA ",USE_CUDA," \nUSE_CONTINUOUS ACTIONS: ",USE_CONTINUOUS)
