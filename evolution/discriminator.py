from evolution import Phenotype
import torch
from evolution import Genome, Linear, Conv2d, Deconv2d
import logging
import torch.autograd as autograd
from torch.autograd import Variable
from .config import config
import util.tensor_constants as tensor_constants
from util import tools
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import numpy as np

logger = logging.getLogger(__name__)


class Discriminator(Phenotype):

    def __init__(self, output_size=1, genome=None, input_shape=None):
        super().__init__(output_size=output_size, genome=genome, input_shape=input_shape)
        self.output_size = output_size

        if genome is None:
            if config.gan.discriminator.fixed:
                self.genome = Genome(random=False, add_layer_prob=0, rm_layer_prob=0, gene_mutation_prob=0,
                                     simple_layers=config.gan.discriminator.simple_layers)
                if not config.gan.discriminator.simple_layers:
                    self.genome.add(Conv2d(8, stride=2, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
                    self.genome.add(Conv2d(16, stride=2, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
                    self.genome.add(Conv2d(32, stride=2, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
                    self.genome.add(Conv2d(64, stride=2, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
                    self.genome.add(Conv2d(128, stride=2, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
                # self.genome.add(Linear(1024, activation_type="ELU"))
            else:
                self.genome = Genome(random=not config.evolution.sequential_layers)
                self.genome.possible_genes = [g for g in self.genome.possible_genes if g[0] != Deconv2d]
                self.genome.add_random_gene()

            if config.gan.type == "gan":
                self.genome.output_genes = [Linear(1, activation_type="Sigmoid", normalize=False)]
            else:
                self.genome.output_genes = [Linear(1, activation_type=None, normalize=False)]

    def forward(self, x):
        out = super().forward(x)
        out = out.view(out.size(0), -1)
        return out

    def train_step(self, G, images):
        """Train the discriminator on real+fake"""
        self.zero_grad()

        #  1A: Train D on real
        real_error, real_decision = self.step_real(images)
        if config.gan.type == "wgan":
            real_error.backward(tensor_constants.ONE)  # compute/store gradients, but don't change params
        elif config.gan.type not in ["rsgan", "rasgan"]:
            real_error.backward()  # compute/store gradients, but don't change params

        #  1B: Train D on fake
        fake_error, fake_data, fake_decision = self.step_fake(G, batch_size=images.size()[0])
        if config.gan.type == "wgan":
            fake_error.backward(tensor_constants.MONE)
        elif config.gan.type not in ["rsgan", "rasgan"]:
            fake_error.backward()

        if config.gan.type == "rsgan":
            labels = tools.cuda(Variable(torch.ones(images.size(0))))
            real_error = self.criterion(real_decision.view(-1) - fake_decision.view(-1), labels)
            real_error.backward()
            fake_error = tools.cuda(torch.FloatTensor([0]))
        elif config.gan.type == "rasgan":
            labels = tools.cuda(Variable(torch.ones(images.size(0))))
            labels_zeros = tools.cuda(Variable(torch.zeros(images.size(0))))
            real_error = (self.criterion(real_decision.view(-1) - torch.mean(fake_decision.view(-1)), labels) + self.criterion(torch.mean(fake_decision.view(-1)) - real_decision.view(-1), labels_zeros))/2
            real_error.backward()
            fake_error = tools.cuda(torch.FloatTensor([0]))

        if config.evolution.fitness.discriminator == "AUC":
            # full_decision = np.concatenate((real_decision.cpu().data.numpy().flatten(), fake_decision.cpu().data.numpy().flatten()))
            # full_labels = np.concatenate((np.ones(real_decision.size()[0]), np.zeros(fake_decision.size()[0])))

            # self.fitness_value -= roc_auc_score(full_labels, full_decision)
            # self.fitness_value -= average_precision_score(full_labels, full_decision)
            # self.fitness_value += 1 - accuracy_score(full_labels, full_decision>0.5)
            # self.fitness_value += np.random.rand()

            self.fitness_value += abs(accuracy_score(np.zeros(fake_decision.size()[0]), fake_decision.cpu().data.numpy().flatten()>0.5) -
                                      accuracy_score(np.ones(real_decision.size()[0]), real_decision.cpu().data.numpy().flatten()>0.5))

        if config.gan.discriminator.use_gradient_penalty:
            gradient_penalty = self.gradient_penalty(images.data, fake_data.data)
            gradient_penalty.backward()

        self.optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
        # Wasserstein distance
        if config.gan.type == "wgan":
            return (real_error - fake_error).item()
        return (real_error + fake_error).item()

    def step_real(self, images):
        real_decision = self(images)

        if config.gan.type in ["wgan", "rsgan", "rasgan"]:
            return real_decision.mean(), real_decision
        elif config.gan.type == "lsgan":
            return 0.5 * torch.mean((real_decision - 1)**2), real_decision

        labels = tools.cuda(Variable(torch.ones(images.size(0))))
        labels = labels * 0.9 if config.gan.label_smoothing else labels
        return self.criterion(real_decision.view(-1), labels), real_decision

    def step_fake(self, G, batch_size):
        gen_input = G.generate_noise(batch_size)
        fake_data = G(gen_input).detach()  # detach to avoid training G on these labels
        fake_decision = self(fake_data)

        if config.gan.type in ["wgan", "rsgan", "rasgan"]:
            return fake_decision.mean(), fake_data, fake_decision
        elif config.gan.type == "lsgan":
            return 0.5 * torch.mean((fake_decision)**2), fake_data, fake_decision

        fake_labels = tools.cuda(Variable(torch.zeros(batch_size)))
        fake_labels = fake_labels + 0.1 if config.gan.label_smoothing else fake_labels
        return self.criterion(fake_decision.view(-1), fake_labels), fake_data, fake_decision

    def eval_step(self, G, images):
        fake_error, _, _ = self.step_fake(G, images.size(0))
        real_error, _ = self.step_real(images)
        return real_error.item() + fake_error.item()

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size()[0]
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = tools.cuda(alpha.expand_as(real_data))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = tools.cuda(self(interpolates))
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=tools.cuda(torch.ones(disc_interpolates.size())),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * config.gan.discriminator.gradient_penalty_lambda

    def fitness(self):
        if config.evolution.fitness.discriminator == "AUC":
            return self.fitness_value
        return super().fitness()
