from .gene import Gene
from ..config import config
import torch.optim as optim
import numpy as np


class Optimizer(Gene):

    def __init__(self):
        super().__init__()
        self.learning_rate = config.optimizer.learning_rate
        self.optimizer = None

    def create_phenotype(self, phenotype):
        if config.optimizer.type == "Adam":
            self.optimizer = optim.Adam(phenotype.parameters(), lr=self.learning_rate, betas=(0, 0.99), weight_decay=config.optimizer.weight_decay)
        elif config.optimizer.type == "SGD":
            self.optimizer = optim.SGD(phenotype.parameters(), nesterov=True, lr=self.learning_rate, momentum=0.95, weight_decay=config.optimizer.weight_decay)
        elif config.optimizer.type == "RMSprop":
            self.optimizer = optim.RMSprop(phenotype.parameters(), lr=self.learning_rate, weight_decay=config.optimizer.weight_decay)
        elif config.optimizer.type == "Adadelta":
            self.optimizer = optim.Adadelta(phenotype.parameters(), weight_decay=config.optimizer.weight_decay)
        else:
            clazz = getattr(optim, config.optimizer.type)
            self.optimizer = clazz(phenotype.parameters(), lr=self.learning_rate, weight_decay=config.optimizer.weight_decay)
        return self.optimizer

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'learning_rate=' + str(self.learning_rate) + ')'

    # def apply_mutation(self):
    #     self.learning_rate += np.random.normal(0.0, self.learning_rate/10)
    #     self.learning_rate = min(max(config.optimizer.learning_rate/10, self.learning_rate), 2*config.optimizer.learning_rate)
