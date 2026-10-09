IMG_SIZE = 224

EMOTION_LABELS = {
    "happy" : 0,
    "sad" : 1,
    "neutral" : 2,
    "angry" :  3,
    "fear" : 4,
}

NEAR_LIMIT= 0.5
FAR_LIMIT = 1.5

#Training const
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 32 

#Paths
TRAINING_DATA_PATH = "data/processed/train"