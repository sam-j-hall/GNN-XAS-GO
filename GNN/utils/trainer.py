import os
import random
import numpy as np
import torch
import torch.nn as nn

def train_model(model, loader, optimizer, device):
    model.train()

    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)

        data_size = batch.spectrum.shape[0] // 200
        batch.spectrum = batch.spectrum.view(data_size, 200)


        loss = nn.MSELoss()(pred, batch.spectrum)

        loss.backward()

        total_loss += loss.item()

        optimizer.step()

        return total_loss
    
def val_model(model, loader, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        data_size = batch.spectrum.shape[0] // 200
        batch.spectrum = batch.spectrum.view(data_size, 200)

        loss = nn.MSELoss()(pred, batch.spectrum)

        total_loss += loss.item()

    return total_loss

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 