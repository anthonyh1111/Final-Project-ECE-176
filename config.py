import torch

# Data
DATA_ROOT = "./data"
NUM_CLASSES = 10
IMAGE_SIZE = 32

# Training (typical values can adjust)
BATCH_SIZE = 64 
NUM_EPOCHS = 10

# regularization training values
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saving path 
MODEL_SAVE_PATH = "baseline_cnn_cifar10.pth"


PRINT_EVERY = 100
