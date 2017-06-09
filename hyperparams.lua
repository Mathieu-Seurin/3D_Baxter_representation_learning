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


--------------   MODELS:
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
