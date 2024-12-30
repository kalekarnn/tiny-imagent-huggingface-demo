from data import get_data_loaders
from model import get_resnet50
from utils import train_model, save_model
from config import *

import torch
import torch.optim as optim
import torch.nn as nn

if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders(BATCH_SIZE)
    model = get_resnet50(num_classes=200)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = train_model(model, train_loader, val_loader, DEVICE, NUM_EPOCHS, criterion, optimizer)
    save_model(model, MODEL_SAVE_PATH)
