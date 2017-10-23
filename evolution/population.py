import numpy as np
from .species import Species
from .config import config


class Population:

    def __init__(self, phenotypes, desired_species, speciation_threshold=config.evolution.speciation.threshold):
        self.species_list = []
        self.speciation_threshold = speciation_threshold
        self.desired_species = desired_species

        if phenotypes is not None:
            self.update(phenotypes)

    def update(self, phenotypes, adjust_threshold=True):
        self.species_list = []
        for p in phenotypes:
            selected_species = self.find_species(p) or Species(speciation_threshold=self.speciation_threshold)
            if len(selected_species) == 0:
                self.species_list.append(selected_species)  # initialize a new population
            selected_species.append(p)

        if adjust_threshold:
            self.adjust_speciation_threshold(phenotypes)
        return self.species_list

    def adjust_speciation_threshold(self, phenotypes, max_runs=1):
        """Adjust the number of species to be around max_species"""
        runs = 0
        while len(self.species_list) != self.desired_species and self.speciation_threshold > 0 and runs < max_runs:
            runs += 1
            self.speciation_threshold += 1 * np.sign(len(self.species_list) - self.desired_species)
            self.speciation_threshold = max(1, self.speciation_threshold)
            self.species_list = self.update(phenotypes, adjust_threshold=False)

    def find_species(self, phenotype):
        for species in self.species_list:
            if species.is_fit(phenotype):
                return species
        return None

    def phenotypes(self):
        return [p for species in self.species_list for p in species]

    def sorted(self):
        return sorted(self.phenotypes(), key=lambda x: x.fitness() or float("inf"))

    def best(self):
        return self.sorted()[0]

    def bests(self, amount):
        return self.sorted()[:amount]

    def best_percent(self, percent):
        sorted_phenotypes = self.sorted()
        return sorted_phenotypes[:int(percent * len(sorted_phenotypes))]
