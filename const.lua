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

--===========================================================
-- CUDA CONSTANTS
--===========================================================
USE_CUDA = true
USE_SECOND_GPU = true

if USE_CUDA and USE_SECOND_GPU then
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

elseif DATA_FOLDER == 'mobileRobot' then

   CLAMP_CAUSALITY = false

   MIN_TABLE = {-10000,-10000} -- for x,y
   MAX_TABLE = {10000,10000} -- for x,y

   DIMENSION_IN = 2

   REWARD_INDICE = 1
   INDICE_TABLE = {1,2} --column indice for coordinate in state file (respectively x,y)

   DEFAULT_PRECISION = 0.1
   FILENAME_FOR_ACTION = "action"
   FILENAME_FOR_STATE = "state"
   FILENAME_FOR_REWARD = "reward"

   SUB_DIR_IMAGE = 'recorded_camera_top'

elseif DATA_FOLDER == 'realBaxterPushingObjects' then  --TODO upload to data_baxter repo
  -- Leni's real Baxter data on  ISIR dataserver. It is named "data_archive_sim_1".
  DEFAULT_PRECISION = 0.1
  -- CLAMP_CAUSALITY = false
  -- MIN_TABLE = {-10000,-10000} -- for x,y
  -- MAX_TABLE = {10000,10000} -- for x,y
  --
  DIMENSION_IN = 3
  REWARD_INDICE = 2
  -- INDICE_TABLE = {1,2} --column indice for coordinate in state file (respectively x,y)
  --
  FILENAME_FOR_ACTION = "action_pushing_object.txt" -- equiv to recorded_button1_is_pressed.txt right now in 3D simulated learning representations
  FILENAME_FOR_STATE = "state_pushing_object"
  FILENAME_FOR_REWARD = "reward_pushing_object"
  --
  SUB_DIR_IMAGE = 'baxter_pushing_objects'

else
  print("No supported data folder provided, please add either of simpleData3D, mobileRobot or Leni's realBaxterPushingObjects")
  os.exit()
end

print("\nUSE_CUDA ",USE_CUDA," \nUSE_CONTINUOUS ACTIONS: ",USE_CONTINUOUS)
