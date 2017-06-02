--DATASETS AVAILABLE:
BABBLING = 'babbling'
MOBILE_ROBOT = 'mobileRobot'
SIMPLEDATA3D = 'simpleData3D'
--================================================

--DATA_FOLDER = BABBLING
DATA_FOLDER = MOBILE_ROBOT

print("============ DATA USED =========\n",
      DATA_FOLDER,
      "\n================================")

--==================================================
-- Hyperparams : Learning rate, batchsize, USE_CUDA etc...
--==================================================

-- Create actions that weren't done by the robot
-- by sampling randomly states (begin point and end point)
-- Cannot be applied in every scenario !!!!
EXTRAPOLATE_ACTION = false

LR=0.001
LR_DECAY = 1e-6

SGD_METHOD = 'adam' -- Can be adam or adagrad
BATCH_SIZE = 2
NB_EPOCHS=15

DATA_AUGMENTATION = 0.01
NORMALIZE_IMAGE = true

COEF_TEMP=1
COEF_PROP=1
COEF_REP=1
COEF_CAUS=1
DIMENSION_OUT=2 --TODO only for mobile data? move to const.lua?


--======================================================
--Continuous actions SETTINGS
--======================================================

USE_CONTINUOUS = false --A switch between discrete and continuous actions (translates into calling getRandomBatchFromSeparateListContinuous instead of getRandomBatchFromSeparateList
ACTION_AMPLITUDE = 0.01
-- The following parameter eliminates the need of finding close enough actions for assessing all priors except for the temporal.one.
-- If the actions are too far away, they will make the gradient 0 and will not be considered for the update rule
GAUSSIAN_SIGMA = 0.1
