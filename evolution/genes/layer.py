import torch.nn as nn
import numpy as np
import uuid
from .gene import Gene
from ..config import config
from ..layers.wscale import WScaleLayer
from ..layers.minibatch_stddev import MinibatchStdDev


class Layer(Gene):
    """Represents a Layer (e.g., FC, conv, deconv) and its
    hyperparameters (e.g., activation function) in the evolving model."""

    # ACTIVATION_TYPES = ["ReLU"]
    ACTIVATION_TYPES = ["ReLU", "LeakyReLU", "ELU", "Sigmoid", "Tanh"]

    def __init__(self, activation_type="random", activation_params={}, normalize=True, use_dropout=False):
        super().__init__()
        self.activation_type = None
        self._original_activation_type = activation_type
        self.activation_params = activation_params
        self.module = None
        self.module_name = None
        self.input_shape = None
        self.final_output_shape = None
        self.next_layer = None
        self.previous_layer = None
        self.normalization = None
        self.normalize = normalize
        self.use_dropout = use_dropout
        self.wscale = None
        self.freezed = False
        self.adjusted = False

    def setup(self):
        if self._original_activation_type != "random":
            self.activation_type = self._original_activation_type
        elif self.activation_type is None:
            self.activation_type = np.random.choice(Layer.ACTIVATION_TYPES)

    def create_phenotype(self, input_shape, final_output_shape):
        self.input_shape = input_shape
        self.final_output_shape = final_output_shape
        self.setup()
        modules = []

        if self.has_minibatch_stddev():
            modules.append(MinibatchStdDev())

        if self.module is None or self.changed() or not config.layer.keep_weights:
            self.normalization = self.create_normalization()
            self.module = self._create_phenotype(self.input_shape)
            if self.has_wscale():
                self.wscale = WScaleLayer(self.module)
            if not self.adjusted:
                self.used = 0
            self.adjusted = False

        modules.append(self.module)
        if self.wscale:
            modules.append(self.wscale)
        if self.activation_type:
            modules.append(getattr(nn, self.activation_type)(**self.activation_params))
        if self.normalization:
            modules.append(self.normalization)
        if self.use_dropout:
            modules.append(nn.Dropout2d(p=0.2))
        return nn.Sequential(*modules)

    def create_normalization(self):
        if self.normalize:
            return self._create_normalization()

    def _create_normalization(self):
        return None

    def _create_phenotype(self, input_size):
        pass

    def changed(self):
        return False

    def has_wscale(self):
        return config.gan.use_wscale and not self.is_last_layer()

    def has_minibatch_stddev(self):
        return config.gan.use_minibatch_stddev and self.is_last_layer() and self.is_linear()

    def is_last_layer(self):
        return self.next_layer is None

    def is_linear(self):
        return True

    def reset(self):
        self.module = None  # reset weights
        self.uuid = uuid.uuid4()  # assign a new uuid

    def named_parameters(self):
        if self.module is None:
            return None
        return self.module.named_parameters()

    def freeze(self):
        if not config.evolution.freeze_when_change:
            return
        print('freeze layer', self.freezed)
        self.freezed = True
        self.module.zero_grad()
        for param in self.module.parameters():
            param.requires_grad = False

    def unfreeze(self):
        if not config.evolution.freeze_when_change:
            return
        print('unfreeze layer', self.freezed)
        self.freezed = False
        for param in self.module.parameters():
            param.requires_grad = True

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'activation_type=' + str(self.activation_type) + ')'
