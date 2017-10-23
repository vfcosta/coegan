from .fid import fid_score
from .fid.inception import InceptionV3
import numpy as np
from util import tools
import logging
import torch
from evolution.config import config


base_fid_statistics = None
inception_model = None
logger = logging.getLogger(__name__)


def initialize_fid(train_loader, sample_size=1000):
    global base_fid_statistics, inception_model
    if inception_model is None:
        inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[config.evolution.fitness.fid_dimension]])
    inception_model = tools.cuda(inception_model)

    if base_fid_statistics is None:
        logger.debug("calculate base fid statistics")
        # TODO see a better way to load images from the train dataset
        train_images = []
        for images, _ in train_loader:
            train_images += list(images.numpy())
            if len(train_images) > sample_size:
                train_images = train_images[:sample_size]
                break
        train_images = np.array(train_images)
        base_fid_statistics = fid_score.calculate_activation_statistics(
            train_images, inception_model, cuda=tools.is_cuda_available(),
            dims=config.evolution.fitness.fid_dimension)
        inception_model.cpu()


def fid(generator, sample_size=1000, noise=None):
    generator.eval()
    with torch.no_grad():
        if noise is None:
            noise = generator.generate_noise(sample_size)
        generated_images = generator(noise.cpu()).detach()
    score = fid_images(generated_images)
    generator.zero_grad()
    return score


def fid_images(generated_images):
    global base_fid_statistics, inception_model
    inception_model = tools.cuda(inception_model)
    m1, s1 = fid_score.calculate_activation_statistics(
        generated_images.data.cpu().numpy(), inception_model, cuda=tools.is_cuda_available(),
        dims=config.evolution.fitness.fid_dimension)
    inception_model.cpu()
    m2, s2 = base_fid_statistics
    ret = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
    return ret


if __name__ == '__main__':
    from evolution.gan_train import GanTrain
    train = GanTrain()
    generators_population, discriminators_population = train.generate_intial_population()
    initialize_fid(train.train_loader)
    score = fid(generators_population.phenotypes()[0])
    logger.debug("fid", score)
