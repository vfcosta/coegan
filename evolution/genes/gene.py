import uuid
import numpy as np


class Gene:
    """Represents an generic gene."""

    def __init__(self):
        self.used = 0
        self.uuid = uuid.uuid4()

    def __repr__(self):
        return self.__class__.__name__

    def mutate(self, probability=0.1):
        if np.random.rand() <= probability:
            self.apply_mutation()
            return True
        return False

    def is_equivalent(self, gene):
        return self.uuid == gene.uuid

    def apply_mutation(self):
        pass
