SUPPORTED_BACKBONE_TYPES = ['resnet']

TRAINING_STATE_SAVE_PATH_FORMAT = 'checkpoints/{}/training_states/{:04d}.state'
MODEL_SAVE_PATH_FORMAT = 'checkpoints/{}/models/{:04d}.pth'

DATASET_ROOT = './dataset/Blur2Vid/RB2V'

CHECKPOINT_PATH = 'checkpoints'

IMG_NORMALIZE_STD = (0.229, 0.224, 0.225)
IMG_NORMALIZE_MEAN = (0.485, 0.456, 0.406)

SUPPORTED_TASKS = ['classification', 'keypoint_detection', 'object_detection']

VAL_SAMPLE_SAVE_PATH = 'checkpoints/{}/val_vis'
TRAIN_SAMPLE_SAVE_PATH = 'checkpoints/{}/train_vis'

SEED = 42
