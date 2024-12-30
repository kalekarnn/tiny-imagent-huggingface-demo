import torch

BATCH_SIZE = 512
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "resnet50_tinyimagenet.pt"
HF_MODEL_REPO = "resnet50-tinyimagenet"