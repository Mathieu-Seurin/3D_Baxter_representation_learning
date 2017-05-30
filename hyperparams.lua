--DATA_FOLDER = 'simpleData3D'
DATA_FOLDER = 'mobileRobot'

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

SGD_METHOD = 'adam' -- Can be adam or adagrad
BATCH_SIZE = 2 -- TRYING TO HAVE BIGGER BATCH
NB_EPOCHS=3

DATA_AUGMENTATION = 0.01
NORMALIZE_IMAGE = true

COEF_TEMP=0.3
COEF_PROP=0.3
COEF_REP=0.3
COEF_CAUS=1
DIMENSION_OUT=2


--======================================================
--Continuous actions SETTINGS
--======================================================
USE_CONTINUOUS = false -- Requires calling getRandomBatchFromSeparateListContinuous instead of getRandomBatchFromSeparateList
ACTION_AMPLITUDE = 0.01
-- The following parameter eliminates the need of finding close enough actions for assessing all priors except for the temporal.one.
-- If the actions are too far away, they will make the gradient 0 and will not be considered for the update rule
GAUSSIAN_SIGMA = 0.1

