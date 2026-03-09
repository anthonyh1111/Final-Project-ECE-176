import torch

# Data
DATA_ROOT = "./data"
NUM_CLASSES = 10
IMAGE_SIZE = 32

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# System
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saving
MODEL_SAVE_PATH = "baseline_cnn_cifar10.pth"

# Logging
PRINT_EVERY = 100
