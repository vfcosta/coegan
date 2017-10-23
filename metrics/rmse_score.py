import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

train_images = []


def initialize(train_loader, sample_size=1000):
    global train_images
    if len(train_images) > 0:
        logger.info("rmse already initialized")
        return
    logger.info(f"initializing rmse score with sample_size={sample_size}")
    for images, _ in train_loader:
        train_images += list(images.numpy())
        if len(train_images) > sample_size:
            train_images = train_images[:sample_size]
            break
    train_images = torch.tensor(train_images)


def rmse(generated_images):
    loss = nn.MSELoss()
    output = torch.sqrt(loss(generated_images, train_images))
    return output.item()
