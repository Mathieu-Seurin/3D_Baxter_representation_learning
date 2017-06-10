-----------------DATASETS AVAILABLE:
BABBLING = 'babbling'
MOBILE_ROBOT = 'mobileRobot'
SIMPLEDATA3D = 'simpleData3D'
PUSHING_BUTTON_AUGMENTED = 'pushingButton3DAugmented'

DATA_FOLDER = MOBILE_ROBOT
--DATA_FOLDER = PUSHING_BUTTON_AUGMENTED
--DATA_FOLDER = BABBLING

print("============ DATA USED =========\n",
                    DATA_FOLDER,
      "\n================================")

--======================================================
--Continuous actions SETTINGS
--======================================================
USE_CONTINUOUS = true --A switch between discrete and continuous actions (translates into calling getRandomBatchFromSeparateListContinuous instead of getRandomBatchFromSeparateList
ACTION_AMPLITUDE = 0.01
-- The following parameter eliminates the need of finding close enough actions for assessing all priors except for the temporal.one.
-- If the actions are too far away, they will make the gradient 0 and will not be considered for the update rule
CONTINUOUS_ACTION_SIGMA = 0.3 -- 0.1 for mobileData plots all concentrated.
--In contiuous actions, we take 2 actions, if they are very similar, the coef factor
--is high (1 if the actions are the same), if not, the coef is 0. You could add a small constraints because the network will see a lot
--of actions that are not similar, so instead of taking '2 random actions', we take '2 random actions, but above a certain similarity threshold'
MAX_DIST_AMONG_ACTIONS_THRESHOLD = 0.5--TODO Find best value

--======================================================
-- Models
--======================================================

INCEPTIONV4 = './models/inceptionFineTunning.lua'
BASE_TIMNET = './models/topUniqueSimplerWOTanh'

--MODEL_ARCHITECTURE_FILE = INCEPTIONV4
MODEL_ARCHITECTURE_FILE = BASE_TIMNET
print("Model :",MODEL_ARCHITECTURE_FILE)

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
NB_EPOCHS=10

DATA_AUGMENTATION = 0.01
NORMALIZE_IMAGE = true

COEF_TEMP=1
COEF_PROP=1
COEF_REP=1
COEF_CAUS=1
DIMENSION_OUT=5
