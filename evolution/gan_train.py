# -*- coding: future_fstrings -*-
#!/usr/bin/env python

# References:
# https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9
# https://www.cs.ucf.edu/~kstanley/neat.html
# https://github.com/CodeReclaimers/neat-python/blob/99da17d4bd71ec97d7f37c9b5df0006c7689a893/neat/reproduction.py

# https://github.com/pytorch/examples/blob/master/dcgan/main.py
import torch
from torch.autograd import Variable
from evolution.discriminator import Discriminator
from evolution.generator import Generator
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from util.stats import Stats
import numpy as np
import util.tools as tools
from random import shuffle
from .population import Population
from tqdm import tqdm
from .config import config
import logging
from util.folder import ImageFolder
from metrics import generative_score

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GanTrain:

    def __init__(self):
        self.stats = Stats()
        self.train_dataset = self.create_dataset()

        train_indexes, validation_indexes = np.split(np.random.permutation(np.arange(len(self.train_dataset))),
                                                     [int(0.9 * len(self.train_dataset))])
        logger.info("train size: %d, validation size: %d" % (len(train_indexes), len(validation_indexes)))
        # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
        train_sampler = torch.utils.data.sampler.SequentialSampler(self.train_dataset)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=config.gan.batch_size,
                                                        sampler=train_sampler, num_workers=0)
        validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(validation_indexes)
        self.validation_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=config.gan.batch_size,
                                                             sampler=validation_sampler)

        self.input_shape = next(iter(self.train_loader))[0].size()[1:]

    @classmethod
    def create_dataset(cls):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        if hasattr(dsets, config.gan.dataset):
            dataset = getattr(dsets, config.gan.dataset)(root=f"./data/{config.gan.dataset}/", train=True,
                                                         download=True, transform=transform)
            if config.gan.dataset_classes:
                indexes = np.argwhere(np.isin(dataset.train_labels, config.gan.dataset_classes))
                dataset.train_data = dataset.train_data[indexes].squeeze()
                dataset.train_labels = np.array(dataset.train_labels)[indexes]
            return dataset
        else:
            return ImageFolder(root=f"./data/{config.gan.dataset}/train", transform=transform)

    def generate_intial_population(self):
        generators = []
        discriminators = []
        for i in range(config.gan.generator.population_size):
            G = Generator(output_size=self.input_shape)
            G.setup()
            generators.append(G)
        for i in range(config.gan.discriminator.population_size):
            D = Discriminator(output_size=1, input_shape=[1]+list(self.input_shape))  # [1] is the batch dimension
            D.setup()
            discriminators.append(D)
        return Population(generators, desired_species=config.evolution.speciation.size),\
               Population(discriminators, desired_species=config.evolution.speciation.size)

    def train_evaluate(self, G, D, train_generator=True, train_discriminator=True, norm_g=1, norm_d=1):
        if G.invalid or D.invalid:  # do not evaluate if G or D are invalid
            logger.warning("invalid D or G")
            return

        torch.cuda.empty_cache()
        n, ng = 0, 0
        G.error = G.error or 0
        D.error = D.error or 0
        g_error = G.error
        d_error = D.error
        d_fitness_value, g_fitness_value = D.fitness_value, G.fitness_value
        G, D = tools.cuda(G), tools.cuda(D)  # load everything on gpu (cuda)
        G.train()
        D.train()
        while n < config.gan.batches_limit:
            for images, _ in self.train_loader:
                # if n==0: print(images[0].mean())
                n += 1
                if n > config.gan.batches_limit:
                    break
                images = tools.cuda(Variable(images))
                if train_discriminator:
                    D.do_train(G, images)
                if train_generator and n % config.gan.critic_iterations == 0:
                    ng += 1
                    G.do_train(D, images)
        if train_discriminator:
            D.error = d_error + (D.error - d_error)/(n*norm_d)
            D.fitness_value = d_fitness_value + (D.fitness_value - d_fitness_value) / (n * norm_d)
            G.fitness_value = g_fitness_value + (G.fitness_value - g_fitness_value) / (n * norm_g)
        if train_generator:
            G.error = g_error + (G.error - g_error)/(ng*norm_g)
        G, D = G.cpu(), D.cpu()  # move variables back from gpu to cpu
        torch.cuda.empty_cache()

    def evaluate_population(self, generators, discriminators, previous_generators, previous_discriminators,
                            best_generators, best_discriminators,
                            evaluation_type=config.evolution.evaluation.type, initial=False):
        """Evaluate the population using all-vs-all pairing strategy"""

        self.train_dataset = torch.utils.data.random_split(self.train_dataset, [len(self.train_dataset)])[0]
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=config.gan.batch_size)

        for i in range(config.evolution.evaluation.iterations):
            shuffle(generators)
            shuffle(discriminators)
            if evaluation_type == "random":
                for D in discriminators:
                    for g in np.random.choice(generators, 2, replace=False):
                        self.train_evaluate(g, D, norm_d=2, norm_g=len(discriminators))
                for G in generators:
                    for d in np.random.choice(discriminators, 2, replace=False):
                        self.train_evaluate(G, d, norm_d=len(generators), norm_g=2)
            elif evaluation_type == "all-vs-all":
                # train all-vs-all in a non-sequential order
                pairs = tools.permutations(generators, discriminators, random=True)
                for g, d in pairs:
                    self.train_evaluate(generators[g], discriminators[d], norm_d=len(generators), norm_g=len(discriminators))
            elif evaluation_type in ["all-vs-best", "all-vs-species-best", "all-vs-kbest"]:
                if config.evolution.evaluation.initialize_all and initial:
                    # as there are no way to determine the best G and D, we rely on all-vs-all for the first evaluation
                    return self.evaluate_population(generators, discriminators,
                                                    previous_generators, previous_discriminators,
                                                    best_generators, best_discriminators, evaluation_type="all-vs-all")
                pairs = tools.permutations(best_generators, discriminators)
                for g, d in pairs:
                    self.train_evaluate(best_generators[g], discriminators[d], norm_d=len(best_generators), norm_g=len(discriminators), train_generator=False)
                pairs = tools.permutations(generators, best_discriminators)
                for g, d in pairs:
                    self.train_evaluate(generators[g], best_discriminators[d], norm_d=len(generators), norm_g=len(best_discriminators), train_discriminator=False)

        if config.evolution.fitness.generator == "FID" or config.stats.calc_fid_score:
            for G in generators:
                G.calc_fid()

        # do not evaluate in the validation data when there is only a single option
        if len(discriminators) == 1 and len(generators) == 1:
            return

        # evaluate in validation (all-vs-best)
        # for D in discriminators:
        #     for G in best_generators:
        #         with torch.no_grad():
        #             self.evaluate_validation(G, D, eval_generator=False)
        # for G in generators:
        #     for D in best_discriminators:
        #         with torch.no_grad():
        #             self.evaluate_validation(G, D, eval_discriminator=False)

    def evaluate_validation(self, G, D, eval_generator=True, eval_discriminator=True, norm_g=1, norm_d=1):
        if G.invalid or D.invalid:  # do not evaluate if G or D are invalid
            logger.warning("invalid D or G")
            return

        if eval_discriminator:
            D.error = 0
        if eval_generator:
            G.error = 0
        n = 0
        G, D = tools.cuda(G), tools.cuda(D)  # load everything on gpu (cuda)
        for images, _ in self.validation_loader:
            images = tools.cuda(Variable(images))
            n += 1
            if eval_discriminator:
                D.do_eval(G, images)
            if eval_generator:
                G.do_eval(D, images)
        if eval_discriminator:
            D.error /= n*norm_d
        if eval_generator:
            G.error /= n*norm_g
        G, D = G.cpu(), D.cpu()  # move variables back from gpu to cpu

    def select(self, population, discard_percent=0, k=config.evolution.tournament_size):
        """Select individuals based on fitness sharing"""

        ### TOURNAMENT TEST
        # population_size = len(population.phenotypes())
        # phenotypes = population.phenotypes()
        # selected = []
        # for i in range(population_size):
        #     p = np.random.choice(phenotypes, 3, replace=False).tolist()
        #     p.sort(key=lambda x: x.fitness())
        #     selected.append([p[0], p[0]])
        # return [selected]
        ###

        population_size = len(population.phenotypes())
        species_selected = []
        species_list = population.species_list
        average_species_fitness_list = []
        for species in species_list[:]:
            species.remove_invalid()  # discard invalid individuals
            if len(species) > 0:
                average_species_fitness_list.append(species.average_fitness())
            else:
                species_list.remove(species)
        total_fitness = np.sum(average_species_fitness_list)

        # initialize raw sizes with equal proportion
        raw_sizes = [population_size / len(species_list)] * len(species_list)
        if total_fitness != 0:
            # calculate proportional sizes when total fitness is not zero
            raw_sizes = [average_species_fitness / total_fitness * population_size
                         for average_species_fitness in average_species_fitness_list]

        sizes = tools.round_array(raw_sizes, max_sum=population_size, invert=True)

        for species, size in zip(species_list, sizes):
            # discard the lowest-performing individuals
            species = species.best_percent(1 - discard_percent)

            # tournament selection inside species
            selected = []

            # ensure that the best was selected
            if config.evolution.speciation.keep_best and size > 0:
                selected.append([species[0]])

            orig_species = list(species)
            for i in range(int(size) - len(selected)):
                parents = []
                for l in range(2):
                    winner = None
                    for j in range(k):
                        random_index = np.random.randint(0, len(species))
                        if winner is None or species[random_index].fitness() < winner.fitness():
                            winner = species[random_index]
                        del species[random_index]  # remove element to emulate draw without replacement
                        if len(species) == 0:  # restore original list when there is no more individuals to draw
                            species = list(orig_species)
                    parents.append(winner)
                    if config.evolution.crossover_rate == 0:
                        # do not draw another individual from the population if there is no probability of crossover
                        parents.append(winner)
                        break
                selected.append(parents)

            species_selected.append(selected)
        return species_selected

    def generate_children(self, species_list, generation):
        # generate child (only mutation for now)
        children = []
        for species in species_list:
            for i, parents in enumerate(species):
                mate = parents[1] if len(parents) > 1 else None
                child = parents[0].breed(mate=mate, skip_mutation=mate is None)  # skip mutation when there is no mate
                child.genome.generation = generation
                children.append(child)
        return children

    def replace_population(self, generators_population, discriminators_population, g_children, d_children):
        elite_d = discriminators_population.best_percent(config.evolution.elitism)
        elite_g = generators_population.best_percent(config.evolution.elitism)

        g_children = sorted(g_children, key=lambda x: x.fitness())
        d_children = sorted(d_children, key=lambda x: x.fitness())

        generators = Population(elite_g + g_children[:len(g_children) - len(elite_g)],
                                desired_species=config.evolution.speciation.size,
                                speciation_threshold=generators_population.speciation_threshold)
        discriminators = Population(elite_d + d_children[:len(d_children) - len(elite_d)],
                                    desired_species=config.evolution.speciation.size,
                                    speciation_threshold=discriminators_population.speciation_threshold)
        return generators, discriminators

    def get_bests(self, population, previous_best):
        if config.evolution.evaluation.type == "all-vs-species-best":
            return [species.best() for species in population.species_list]
        elif config.evolution.evaluation.type == "all-vs-best":
            return (population.bests(1) + previous_best)[:config.evolution.evaluation.best_size]
        elif config.evolution.evaluation.type == "all-vs-kbest":
            return population.bests(config.evolution.evaluation.best_size)

    def start(self):
        if config.evolution.fitness.generator == "FID" or config.stats.calc_fid_score:
            generative_score.initialize_fid(self.train_loader, sample_size=config.evolution.fitness.fid_sample_size)

        generators_population, discriminators_population = self.generate_intial_population()
        # initialize best_discriminators and best_generators with random individuals
        best_discriminators = list(np.random.choice(discriminators_population.phenotypes(), config.evolution.evaluation.best_size, replace=False))
        best_generators = list(np.random.choice(generators_population.phenotypes(), config.evolution.evaluation.best_size, replace=False))
        # initial evaluation
        self.evaluate_population(generators_population.phenotypes(), discriminators_population.phenotypes(),
                                 generators_population, discriminators_population,
                                 best_generators, best_discriminators, initial=True)
        # store best individuals
        best_discriminators = self.get_bests(discriminators_population, best_discriminators)
        best_generators = self.get_bests(generators_population, best_generators)
        generation = 0

        for generation in tqdm(range(config.evolution.max_generations-1)):
            self.stats.generate(self.input_shape, generators_population, discriminators_population,
                                generation, config.evolution.max_generations, self.train_loader, self.validation_loader)
            # select parents for reproduction
            g_parents = self.select(generators_population)
            d_parents = self.select(discriminators_population)
            # apply variation operators (only mutation for now)
            g_children = self.generate_children(g_parents, generation)

            # limit the number of layers in D's to the max layers among G's
            max_layers_g = max([len(gc.genome.genes) for gc in g_children])
            for s in d_parents:
                for dp in s:
                    dp[0].genome.max_layers = max_layers_g

            d_children = self.generate_children(d_parents, generation)
            # evaluate the children population and the best individuals (when elitism is being used)
            logger.debug(f"[generation {generation}] evaluate population")
            self.evaluate_population(g_children, d_children, generators_population, discriminators_population, best_generators, best_discriminators)
            # store best of generation in coevolution memory
            best_discriminators = self.get_bests(discriminators_population, best_discriminators)
            best_generators = self.get_bests(generators_population, best_generators)
            # generate a new population based on the fitness of the children and elite individuals
            generators_population, discriminators_population = self.replace_population(generators_population,
                                                                                        discriminators_population,
                                                                                        g_children, d_children)
        # stats for last generation
        self.stats.generate(self.input_shape, generators_population, discriminators_population,
                            generation+1, config.evolution.max_generations, self.train_loader, self.validation_loader)
