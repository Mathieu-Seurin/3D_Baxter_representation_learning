-----------------DATASETS AVAILABLE:  (in order of model robustness and reliability so far)
--=================================================
MOBILE_ROBOT = 'mobileRobot'
----Baxter datasets:
SIMPLEDATA3D = 'simpleData3D' --  oldest simplest dataset
PUSHING_BUTTON_AUGMENTED = 'pushingButton3DAugmented'
BABBLING = 'babbling'
STATIC_BUTTON_SIMPLEST = 'staticButtonSimplest'
--!!! AVOID SETTING IT HERE FOR INCONSISTENCIES< SET VIA COMMAND LINE !!!
--DATA_FOLDER = MOBILE_ROBOT
--DATA_FOLDER = STATIC_BUTTON_SIMPLEST --PUSHING_BUTTON_AUGMENTED
--DATA_FOLDER = BABBLING

--================ MODEL USED =====================
--=================================================
INCEPTIONV4 = './models/inceptionFineTunning' --finetuned trained model

RESNET = './models/resnet'  --finetuned trained model

RESNET_VERSION = 18 --34 or 50 maybe
FROZEN_LAYER = 3 --the number of layers that don't learn at all (i.e., their learning_rate=0)

AENET = './models/autoencoder_conv'
BASE_TIMNET = './models/topUniqueSimplerWOTanh'--ImageNet-inspired Convolutional network with ReLu. This is the only model that should be used with learn_autoencoder, not in regular training in script.lua
--otherwise, we get:  /home/gpu_center/torch/install/bin/lua: imagesAndReprToTxt.lua:53: bad argument #1 to 'size' (out of range)

--MODEL_ARCHITECTURE_FILE = INCEPTIONV4 --Too big
-- MODEL_ARCHITECTURE_FILE = BASE_TIMNET--without last layer as Tanh
MODEL_ARCHITECTURE_FILE = RESNET


--==================================================
-- Hyperparams : Learning rate, batchsize, USE_CUDA etc...
--==================================================

-- EXTRAPOLATE_ACTION, if true, selects actions that weren't done by the robot
-- by randomly sampling states (begin point and end point). CLAMP_CAUSALITY,
-- on the contrary, takes the next consecutive action
-- Cannot be applied in every scenario !!!!

EXTRAPOLATE_ACTION = false
EXTRAPOLATE_ACTION_CAUS = false
--TODO shall it be true for continuous actions too always?
-- Always : i don't think so, but trying to see if it works better with it, why not


LR=0.001
LR_DECAY = 3*1e-6

SGD_METHOD = 'adam' -- Can be adam or adagrad
BATCH_SIZE = 16
NB_EPOCHS= 2

DATA_AUGMENTATION = 0.01
NORMALIZE_IMAGE = true

COEF_TEMP=1
COEF_PROP=1
COEF_REP=1
COEF_CAUS=1
DIMENSION_OUT=2
