from evolution import Phenotype
import torch
from torch.autograd import Variable
from evolution import Genome, Linear, Layer, Layer2D, Conv2d, Deconv2d
import logging
from .config import config
import util.tensor_constants as tensor_constants
from util.inception_score import inception_score
from util import tools
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from metrics import generative_score
import os
from metrics import rmse_score


logger = logging.getLogger(__name__)


class Generator(Phenotype):

    fid_noise = None

    def __init__(self, output_size=(1, 28, 28), genome=None, input_shape=(1, 1, 10, 10)):
        super().__init__(output_size=output_size, genome=genome, input_shape=input_shape)
        self.noise_size = int(np.prod(self.input_shape[1:]))
        self.inception_score_mean = 0
        self.fid_score = None
        self.rmse_score = None

        if genome is None:
            if config.gan.generator.fixed:
                self.genome = Genome(random=False, add_layer_prob=0, rm_layer_prob=0, gene_mutation_prob=0,
                                     simple_layers=config.gan.generator.simple_layers, linear_at_end=False)
                self.genome.add(Linear(4*int(np.prod(output_size)), activation_type="ReLU"))
                # self.genome.add(Linear(4*int(np.prod(output_size)), activation_type="LeakyReLU"))
                if not config.gan.generator.simple_layers:
                    self.genome.add(Deconv2d(128, activation_type="ReLU"))
                    self.genome.add(Deconv2d(64, activation_type="ReLU"))
                    self.genome.add(Deconv2d(32, activation_type="ReLU"))
                    self.genome.add(Deconv2d(16, activation_type="ReLU"))
                    # self.genome.add(Deconv2d(8, activation_type="ReLU"))
            else:
                self.genome = Genome(random=not config.evolution.sequential_layers, linear_at_end=False)
                self.genome.possible_genes = [g for g in self.genome.possible_genes if g[0] != Conv2d]
                # IMPORTANT: the performance without a liner layer is pretty bad
                self.genome.add(Linear(512))
                # self.genome.add_random_gene()
            if config.gan.generator.simple_layers:
                # self.genome.output_genes = [Deconv2d(output_size[0], activation_type="Tanh")]
                self.genome.output_genes = [Linear(int(np.prod(output_size)), activation_type="Tanh", normalize=False)]
            else:
                self.genome.output_genes = [Deconv2d(output_size[0], activation_type="Tanh", normalize=False)]

    def forward(self, x):
        out = super().forward(x)
        if out is not None and len(out.size()) == 2:
            out = out.view(out.size(0), *self.output_size)
        return out

    def train_step(self, D, images):
        self.inception_score_mean = 0
        batch_size = images.size(0)
        # 2. Train G on D's response (but DO NOT train D on these labels)
        self.zero_grad()

        error, decision = self.step(D, batch_size)
        if config.gan.type == "wgan":
            error.backward(tensor_constants.ONE)
        elif config.gan.type == "rsgan":
            real_decision = D(images)
            labels = tools.cuda(Variable(torch.ones(images.size(0))))
            error = self.criterion(decision.view(-1) - real_decision.view(-1), labels)
            error.backward()
        elif config.gan.type == "rasgan":
            real_decision = D(images)
            labels = tools.cuda(Variable(torch.ones(images.size(0))))
            labels_zeros = tools.cuda(Variable(torch.zeros(images.size(0))))
            error = (self.criterion(real_decision.view(-1) - torch.mean(decision.view(-1)), labels_zeros) + self.criterion(torch.mean(decision.view(-1)) - real_decision.view(-1), labels))/2
            error.backward()
        else:
            error.backward()

        if config.evolution.fitness.generator == "AUC":
            labels = np.ones(images.size(0))
            self.fitness_value += 1 - accuracy_score(labels, decision.cpu()>0.5)

        self.optimizer.step()  # Only optimizes G's parameters
        if config.gan.type == "wgan":
            return error.item()

        return error.item()

    def step(self, D, batch_size, gen_input=None):
        if gen_input is None:
            gen_input = self.generate_noise(batch_size)
        real_labels = tools.cuda(Variable(torch.ones(batch_size)))
        fake_data = self(gen_input)
        fake_decision = D(fake_data)

        if config.gan.type in ["wgan", "rsgan", "rasgan"]:
            return fake_decision.mean(), fake_decision
        elif config.gan.type == "lsgan":
            return 0.5 * torch.mean((fake_decision - 1) ** 2), fake_decision

        real_labels = real_labels * 0.9 if config.gan.label_smoothing else real_labels
        return self.criterion(fake_decision.view(-1), real_labels), fake_decision

    def eval_step(self, D, images):
        error, decision = self.step(D, images.size(0))
        return error.item()

    def generate_noise(self, batch_size, volatile=False):
        with torch.set_grad_enabled(not volatile):
            gen_input = tools.cuda(Variable(torch.randn(batch_size, self.noise_size)))
        return gen_input.view([batch_size] + list(self.input_shape[1:]))

    def inception_score(self, batch_size=2, splits=10):
        """Computes the inception score of the generated images
        n -- amount of generated images
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
        """
        generated_images = self(self.generate_common_noise()).detach()
        self.inception_score_mean, _ = inception_score(generated_images,
                                                       batch_size=batch_size, resize=True, splits=splits)
        return self.inception_score_mean

    def calc_rmse_score(self):
        generated_images = self(self.generate_common_noise()).detach()
        self.rmse_score = rmse_score.rmse(generated_images)

    def generate_common_noise(self, noise_path='generator_noise.pt'):
        """Generate a noise to be used as base for comparisons"""
        if os.path.isfile(noise_path) and Generator.fid_noise is None:
            Generator.fid_noise = torch.load(noise_path)
            logger.info(f"generator noise loaded from file with shape {Generator.fid_noise.shape}")
            if Generator.fid_noise.shape[0] != config.evolution.fitness.fid_sample_size:
                logger.info(f"discard loaded generator noise because the sample size is different: {config.evolution.fitness.fid_sample_size}")
                Generator.fid_noise = None
        if Generator.fid_noise is None:
            Generator.fid_noise = self.generate_noise(config.evolution.fitness.fid_sample_size).cpu()
            torch.save(Generator.fid_noise, noise_path)
            logger.info(f"generator noise saved to file with shape {Generator.fid_noise.shape}")
        return Generator.fid_noise

    def calc_fid(self):
        self.fid_score = generative_score.fid(self, noise=self.generate_common_noise())

    def fitness(self):
        if config.evolution.fitness.generator == "FID":
            return self.fid_score
        if config.evolution.fitness.generator == "AUC":
            return self.fitness_value
        return super().fitness()
