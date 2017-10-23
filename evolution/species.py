import numpy as np


class Species(list):

    def __init__(self, speciation_threshold=2):
        self.speciation_threshold = speciation_threshold

    def is_fit(self, individual):
        return individual.genome.distance(self[0].genome) < self.speciation_threshold

    def average_fitness(self):
        return np.mean([self.fitness(p) for p in self])

    def sorted(self):
        return sorted(self, key=lambda x: self.fitness(x))

    def fitness(self, phenotype):
        """Calculates fitness adjusted by speciation"""
        return phenotype.fitness() / len(self)

    def best(self):
        return self.sorted()[0]

    def best_percent(self, percent):
        sorted_individuals = self.sorted()
        return sorted_individuals[:int(percent * len(sorted_individuals))]

    def remove_invalid(self):
        for i in self[:]:
            if i.invalid:
                self.remove(i)
