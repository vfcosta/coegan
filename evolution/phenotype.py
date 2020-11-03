import torch
from torch.autograd import Variable
import torch.nn as nn
from .genes import Linear, Layer, Deconv2d
from .layers.reshape import Reshape
import numpy as np
import copy
import traceback
from .config import config
import logging
import json

logger = logging.getLogger(__name__)


class Phenotype(nn.Module):

    def forward(self, x):
        try:
            out = self.model(x)
            return out
        except Exception as err:
            logger.error(err)
            traceback.print_exc()
            self.optimizer = None
            self.invalid = True

    def __init__(self, output_size, genome=None, input_shape=None):
        super().__init__()
        self.genome = genome
        self.optimizer = None
        self.model = None
        self.output_size = output_size
        self.error = None
        self.fitness_value = 0
        self.invalid = False
        self.input_shape = input_shape
        self.trained_samples = 0
        self.random_fitness = None
        if config.gan.type in ["rsgan", "rasgan"]:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCELoss()

    def breed(self, mate=None, skip_mutation=False):
        mate_genome = mate.genome if mate else None
        genome = self.genome.breed(skip_mutation=skip_mutation, mate=mate_genome)
        p = self.__class__(output_size=self.output_size, genome=genome, input_shape=self.input_shape)
        try:
            p.setup()
            self.copy_to(p)
            if mate:
                mate.copy_to(p)
        except Exception as err:
            logger.error(err)
            traceback.print_exc()
            logger.debug(genome)
            p.optimizer = None
            p.invalid = True
            p.error = 100
            if not skip_mutation or mate is not None:
                logger.debug("fallback to parent copy")
                return self.breed(mate=None, skip_mutation=True)
        return p

    def setup(self):
        # create some input data
        with torch.no_grad():
            x = Variable(torch.randn(self.input_shape[0], int(np.prod(self.input_shape[1:]))))
        x = x.view(self.input_shape)  # convert to the same shape as input
        self.create_model(x)

    def copy_to(self, target):
        """
        Copy the phenotype parameters to the target.
        This copy will keep the parameters that match in size from the optimizer.
        """
        target.trained_samples = self.trained_samples
        if not config.optimizer.copy_optimizer_state:
            return

        old_state_dict = self.optimizer.state_dict()
        if len(old_state_dict['state']) == 0:
            return  # there is no state to copy

        # this causes a memory leak with Adam optimizer
        for gene in target.genome.genes:
            old_gene = self.genome.get_gene_by_uuid(gene.uuid)
            if old_gene is None:
                continue
            for (_, param), (_, old_param) in zip(gene.named_parameters(), old_gene.named_parameters() or {}):
                if id(old_param) not in old_state_dict['state']:
                    continue
                old_state = old_state_dict['state'][id(old_param)]
                if ('momentum_buffer' in old_state and old_state['momentum_buffer'].size() == param.data.size()) or \
                        ('exp_avg' in old_state and old_state['exp_avg'].size() == param.data.size()) or \
                        ('square_avg' in old_state and old_state['square_avg'].size() == param.data.size()):
                    target.optimizer.state[param] = copy.deepcopy(old_state)

    def do_train(self, phenotype, images):
        if phenotype.invalid:
            return
        if self.invalid:
            self.error = 100
            return
        try:
            if self.error is None:
                self.error = 0
            self.error += self.train_step(phenotype, images)
            self.trained_samples += len(images)
            self.genome.increase_usage_counter()
        except Exception as err:
            traceback.print_exc()
            logger.error(err)
            logger.error(self.model)
            self.error += 100  # penalty for invalid genotype

    def do_eval(self, phenotype, images):
        if self.invalid:
            self.error = 100
            return
        self.error = self.error or 0
        self.error += self.eval_step(phenotype, images)

    def create_model(self, input_data):
        self.input_shape = input_data.size()
        if "model" not in self._modules:
            self.model = self.transform_genotype(input_data)
        if self.optimizer is None:
            self.optimizer = self.genome.optimizer_gene.create_phenotype(self)

    def transform_genotype(self, input_data):
        """Generates a generic model using pytorch."""
        layers = []

        genes = list(self.genome.genes)
        genes += self.genome.output_genes  # add output layers

        # count how many deconvs exists in the model
        deconv_layers = list(filter(lambda x: isinstance(x, Deconv2d), genes))
        n_deconv2d = len(deconv_layers)
        if deconv_layers:
            # consider the first deconv as a layer that will increase out channels but not the image size
            n_deconv2d -= 1

        has_new_gene = len([g for g in genes if g.used == 0]) > 0

        # iterate over genes to create a pytorch sequential model
        for i, gene in enumerate(genes):
            # TODO: move this code into the Genome class
            if i + 1 < len(genes):  # link the current gene with the next
                gene.next_layer = genes[i+1]
            if i > 0:
                gene.previous_layer = genes[i-1]

            next_input_size, next_input_shape = self.calc_output_size(layers, input_data)

            # adjust shape for linear layer
            if gene.is_linear() and len(next_input_shape) > 2:
                layers.append(Reshape((-1, next_input_size)))

            # adjust out_features of the last linear layer
            if isinstance(gene, Linear) and gene.is_last_linear():
                if isinstance(self.output_size, int):
                    gene.out_features = self.output_size
                else:
                    div = 2**n_deconv2d
                    gene.out_features = div*div * self.output_size[0] * int(np.round(self.output_size[1]/div)) * int(np.round(self.output_size[2]/div))

            # adjust shape for 2d layer
            if not gene.is_linear() and len(next_input_shape) == 2:
                d = max(1, int(next_input_size//np.prod(self.output_size[1:])))  # calc the multiple of output shape (e.g., dx28x28)
                # adjust shape based on the next layers
                div = 2**n_deconv2d
                next_input_shape = (-1, d*div*div, int(np.round(self.output_size[1]/div)), int(np.round(self.output_size[2]/div)))
                layers.append(Reshape(next_input_shape))

            new_layer = gene.create_phenotype(next_input_shape, self.output_size)
            if gene.used > 0 and has_new_gene:
                gene.freeze()
            else:
                gene.unfreeze()
            gene.module_name = "model.%d" % len(layers)
            layers.append(new_layer)

        return nn.Sequential(*layers)

    def calc_output_size(self, layers, input_data):
        current_model = nn.Sequential(*layers)
        current_model.eval()
        forward_pass = current_model(input_data)
        # return the product of the vector array (ignoring the batch size)
        return int(np.prod(forward_pass.size()[1:])), forward_pass.size()

    def set_requires_grad(self, value):
        for p in self.parameters():
            p.requires_grad = value

    def fitness(self):
        if config.evolution.fitness.generator == "random":
            if self.random_fitness is None:
                self.random_fitness = np.random.rand()
            return self.random_fitness
        return self.error

    def save(self, path):
        torch.save(self.cpu(), path)

    def valid(self):
        return not self.invalid and self.error is not None

    @classmethod
    def load(cls, path):
        return torch.load(path, map_location="cpu")

    def __repr__(self):
        output_genes_str = " -> ".join([str(g) for g in self.genome.output_genes])
        return self.__class__.__name__ + f"(genome={self.genome}, output_layers={output_genes_str})"

    def to_json(self):
        """Create a json representing the model"""
        ret = []
        for gene in self.genome.genes + self.genome.output_genes:
            d = dict(gene.__dict__)
            del d["uuid"], d["module"], d["next_layer"], d["previous_layer"], d["normalization"], d["wscale"]
            ret.append({
                "type": gene.__class__.__name__,
                "wscale": gene.has_wscale(),
                "minibatch_stddev": gene.has_minibatch_stddev(),
                "normalization": gene.normalization.__class__.__name__ if gene.normalization is not None else None,
                **d
            })
        return json.dumps(ret)
