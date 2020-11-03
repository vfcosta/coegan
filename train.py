import torch
from evolution.gan_train import GanTrain
# import better_exceptions; better_exceptions.hook()
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    logger.info("CUDA device detected!")
else:
    logger.info("CUDA device not detected!")

if __name__ == "__main__":
    GanTrain().start()
