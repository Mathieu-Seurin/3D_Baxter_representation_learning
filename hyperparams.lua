-----------------DATASETS AVAILABLE:  (in order of model robustness and reliability so far)
--=================================================
MOBILE_ROBOT = 'mobileRobot'
----Baxter datasets:
SIMPLEDATA3D = 'simpleData3D' --  oldest simplest dataset
PUSHING_BUTTON_AUGMENTED = 'pushingButton3DAugmented'
BABBLING = 'babbling'

DATA_FOLDER = MOBILE_ROBOT
--DATA_FOLDER = PUSHING_BUTTON_AUGMENTED
--DATA_FOLDER = BABBLING

--================ MODEL USED =====================
--=================================================
INCEPTIONV4 = './models/inceptionFineTunning' --finetuned trained model

RESNET = './models/resnet'
RESNET_VERSION = 18 --34 or 50 maybe
FROZEN_LAYER = 3 --the number of layers that don't learn at all (i.e., their learning_rate=0)

BASE_TIMNET = './models/topUniqueSimplerWOTanh'

--MODEL_ARCHITECTURE_FILE = INCEPTIONV4 --Too big
MODEL_ARCHITECTURE_FILE = BASE_TIMNET--without last layer as Tanh
--MODEL_ARCHITECTURE_FILE = RESNET
print("Model :",MODEL_ARCHITECTURE_FILE)

--======================================================
--Continuous actions SETTINGS
--======================================================
USE_CONTINUOUS = false --A switch between discrete and continuous actions (translates into calling getRandomBatchFromSeparateListContinuous instead of getRandomBatchFromSeparateList
ACTION_AMPLITUDE = 0.01
-- The following parameter eliminates the need of finding close enough actions for assessing all priors except for the temporal.one.
-- If the actions are too far away, they will make the gradient 0 and will not be considered for the update rule
CONTINUOUS_ACTION_SIGMA = 0.6
--In contiuous actions, we take 2 actions, if they are very similar, the coef factor
--is high (1 if the actions are the same), if not, the coef is close to 0. We add a constraint with the method
--action_vectors_are_similar_enough to impose a cosine distance constraint when comparing actions, because the network will see a lot
--of actions that are not similar, so instead of taking '2 random actions', we take '2 random actions, but above a certain similarity threshold'.
MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD = 0.4
-- TODO shall it be different for each dataset depending on the variance of the input state space?
--If so, What is a good proxy  parameter to set it?

--==================================================
-- Hyperparams : Learning rate, batchsize, USE_CUDA etc...
--==================================================

-- EXTRAPOLATE_ACTION, if true, selects actions that weren't done by the robot
-- by randomly sampling states (begin point and end point). CLAMP_CAUSALITY,
-- on the contrary, takes the next consecutive action
-- Cannot be applied in every scenario !!!!
EXTRAPOLATE_ACTION = false

LR=0.0001
LR_DECAY = 1e-6

SGD_METHOD = 'adam' -- Can be adam or adagrad
BATCH_SIZE = 5
NB_EPOCHS=10

DATA_AUGMENTATION = 0.01
NORMALIZE_IMAGE = true

COEF_TEMP=1
COEF_PROP=1
COEF_REP=1
COEF_CAUS=1
DIMENSION_OUT=2


if DATA_FOLDER ~= BABBLING then
    PRIORS_TO_APPLY ={{"Prop","Temp","Caus","Rep"}}
else
    -- Causality needs at least 2 different values of reward and in sparse dataset such as babbling_1, this does not occur always
    PRIORS_TO_APPLY ={{"Rep","Prop","Temp"}}
    print('WARNING: Until no more than one reward is available, Causality prior will be ignored for dataset '..BABBLING)
end
