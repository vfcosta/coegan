import torch
from torch.autograd import Variable
import unittest
from evolution import Genome, Layer, Linear, Conv2d, Phenotype, Dropout, Deconv2d
import copy
import numpy as np
import json
from evolution.config import config


class TestEvolution(unittest.TestCase):

    def setUp(self):
        self.genome = Genome()
        self.phenotype = Phenotype(1, self.genome)
        config.optimizer.copy_optimizer_state = True
        config.optimizer.type = "Adam"
        config.evolution.sequential_layers = True

    def evaluate_model(self, input_shape):
        x = Variable(torch.randn(input_shape[0], int(np.prod(input_shape[1:]))))  # create some input data
        x = x.view(input_shape)  # convert to 4d (batch_size, channels, w, h)
        model = self.phenotype.transform_genotype(x)
        return model(x)

    def train_step(self, phenotype, x):
        out = phenotype.model(x)
        if out.view(-1).size(0) == out.size(0):
            error = phenotype.criterion(out.view(-1), Variable(torch.ones(out.size(0))))
            error.backward()
        else:
            print("out sizes don't match", out.view(-1).size(0), out.size(0))
        phenotype.optimizer.step()

    def test_valid_genotype(self):
        for i in range(3):
            self.genome.add(Linear(5))
        x = Variable(torch.randn(5, 64))  # create some input data
        model = self.phenotype.transform_genotype(x)
        self.assertEqual(3, len(model))

    def test_valid_phenotype(self):
        self.genome.add(Linear(32))
        self.genome.add(Linear(16))
        self.genome.add(Linear(1))
        out = self.evaluate_model([5, 64])
        self.assertEqual((5, 1), out.shape)

    def test_valid_phenotype_activation(self):
        self.genome.add(Linear(32))
        self.genome.add(Linear())
        self.genome.add(Linear(16))
        x = Variable(torch.randn(5, 64))  # create some input data
        self.phenotype.output_size = 16
        model = self.phenotype.transform_genotype(x)
        out = model(x)  # run pytorch module to check if everything is ok
        self.assertEqual((5, 16), out.shape)

    def test_adjust_last_linear(self):
        self.genome.linear_at_end = False
        self.genome.add(Linear(16))
        x = Variable(torch.randn(5, 64))  # create some input data
        model = self.phenotype.transform_genotype(x)
        out = model(x)  # run pytorch module to check if everything is ok
        self.assertEqual((5, 1), out.shape)
        self.genome.add(Linear(4))
        self.phenotype.transform_genotype(x)
        self.assertEqual(16, self.genome.genes[0].out_features)

    def test_valid_phenotype_activation_leakyrelu(self):
        self.genome.add(Linear(32))
        self.genome.add(Linear(activation_type="LeakyReLU", activation_params={"negative_slope": 0.2, "inplace": True}))
        self.genome.add(Linear(16))
        x = Variable(torch.randn(5, 64))  # create some input data
        self.phenotype.output_size = 16
        model = self.phenotype.transform_genotype(x)
        out = model(x)  # run pytorch module to check if everything is ok
        self.assertEqual((5, 16), out.shape)

    def test_add_random_gene(self):
        for i in range(10):
            self.genome.add_random_gene()
        self.genome.add(Linear(1))
        x = Variable(torch.randn(5, 1*32*32))  # create some input data
        x = x.view(5, 1, 32, 32)
        model = self.phenotype.transform_genotype(x)
        out = model(x)  # run pytorch module to check if everything is ok
        self.assertEqual((5, 1), out.shape)

    def test_reuse_weights(self):
        x = Variable(torch.randn(5, 64))
        self.genome.add(Conv2d(2))
        self.genome.add(Linear(16))
        x = x.view(5, 1, 8, 8)
        self.phenotype.create_model(x)
        self.genome.rm_layer_prob = 0
        self.genome.add_layer_prob = 0
        self.genome.gene_mutation_prob = 0
        phenotype2 = self.phenotype.breed()
        self.assertTrue(self.genome.genes[0].module.weight.equal(phenotype2.genome.genes[0].module.weight))
        self.assertTrue(self.genome.genes[1].module.weight.equal(phenotype2.genome.genes[1].module.weight))

    def test_breed_phenotype_with_new_layer(self):
        x = Variable(torch.randn(5, 64)).view(5, 1, 8, 8)
        self.genome.add(Conv2d(2))
        self.genome.add(Linear(16))
        self.genome.output_genes.append(Linear(activation_type='Sigmoid'))
        self.phenotype.create_model(x)
        self.train_step(self.phenotype, x)
        self.genome.add_layer_prob = 1
        self.genome.rm_layer_prob = 0
        self.genome.gene_mutation_prob = 0
        phenotype2 = self.phenotype.breed()
        old_state = self.phenotype.optimizer.state[self.phenotype.optimizer.param_groups[0]['params'][0]]
        new_state = phenotype2.optimizer.state[phenotype2.optimizer.param_groups[0]['params'][0]]
        self.assertTrue(old_state['exp_avg'].equal(new_state['exp_avg']))
        self.assertIsNot(old_state['exp_avg'], new_state['exp_avg'])
        self.train_step(phenotype2, x)

    def test_not_copy_optimizer_new_module(self):
        x = Variable(torch.randn(5, 64)).view(5, 1, 8, 8)
        self.genome.add(Conv2d(2))
        self.genome.add(Linear(16))
        self.genome.output_genes.append(Linear(activation_type='Sigmoid'))
        self.phenotype.create_model(x)
        self.train_step(self.phenotype, x)
        self.genome.add_layer_prob = 0
        conv2d_module = self.genome.genes[0].module
        self.genome.genes[0].module = None
        phenotype2 = self.phenotype.breed()
        old_state = self.phenotype.optimizer.state[self.phenotype.optimizer.param_groups[0]['params'][0]]
        new_state = phenotype2.optimizer.state[phenotype2.optimizer.param_groups[0]['params'][0]]
        self.assertEqual(0, phenotype2.genome.genes[0].used)
        self.assertFalse(conv2d_module.weight.equal(phenotype2.genome.genes[0].module.weight))
        self.assertIn('exp_avg', old_state)
        self.assertNotIn('exp_avg', new_state)

    def test_not_share_reused_weights(self):
        x = Variable(torch.randn(5, 64)).view(5, 1, 8, 8)
        self.genome.add(Conv2d(2))
        self.genome.add(Linear(16))
        self.genome.output_genes.append(Linear(activation_type='Sigmoid'))
        self.phenotype.create_model(x)
        self.train_step(self.phenotype, x)
        self.genome.add_layer_prob = 0
        self.genome.gene_mutation_prob = 0
        phenotype2 = self.phenotype.breed(skip_mutation=True)
        self.train_step(phenotype2, x)
        self.assertIsNot(self.genome.genes[0].module, phenotype2.genome.genes[0].module)
        self.assertFalse(self.genome.genes[0].module.weight.equal(phenotype2.genome.genes[0].module.weight))
        self.assertFalse(self.genome.genes[1].module.weight.equal(phenotype2.genome.genes[1].module.weight))

    def test_reset_module_when_changed(self):
        x = Variable(torch.randn(5, 64)).view(5, 1, 8, 8)
        self.genome.add(Conv2d(2))
        self.genome.add(Linear(16))
        self.phenotype.transform_genotype(x)
        genome = copy.deepcopy(self.genome)
        genome.genes.insert(0, Conv2d(3))
        genome.add(Linear(32))
        self.phenotype.genome = genome
        self.phenotype.transform_genotype(x)
        self.assertFalse(self.genome.genes[0].module.weight.equal(genome.genes[0].module.weight))
        self.assertFalse(self.genome.genes[1].module.weight.equal(genome.genes[2].module.weight))

    def test_conv2d_phenotype(self):
        self.genome.add(Conv2d(3))
        self.genome.add(Conv2d(6))
        self.genome.add(Linear(1))
        x = Variable(torch.randn(5, 144))  # create some input data
        x = x.view(5, 1, 12, 12)  # convert to 4d (batch_size, channels, w, h)
        model = self.phenotype.transform_genotype(x)
        out = model(x)  # run pytorch module to check if everything is ok
        self.assertEqual((5, 1), out.shape)

    def test_linear_after_conv2d(self):
        self.genome.add(Conv2d(1))
        self.genome.add(Linear(32))
        self.genome.add(Conv2d(3))
        self.assertEqual([Conv2d, Conv2d, Linear], [gene.__class__ for gene in self.genome.genes])
        self.evaluate_model([5, 1, 12, 12])

    def test_first_linear_after_conv2d(self):
        self.genome.add(Linear(32))
        self.genome.add(Conv2d(3))
        self.assertEqual([Conv2d, Linear], [gene.__class__ for gene in self.genome.genes])
        self.evaluate_model([5, 3, 5, 5])

    def test_complex_graph(self):
        self.genome.add(Linear(32))
        self.genome.add(Linear(activation_type="ReLU"))
        self.genome.add(Conv2d(3))
        self.genome.add(Dropout())
        self.assertEqual([Conv2d, Linear, Linear, Dropout], [gene.__class__ for gene in self.genome.genes])
        self.evaluate_model([5, 3, 5, 5])

    def test_complex_graph2(self):
        self.genome.add(Conv2d(1))
        self.genome.add(Linear(128))
        self.evaluate_model([5, 1, 28, 28])

    def test_zero_output(self):
        self.genome.add(Conv2d(1, kernel_size=3))
        self.genome.add(Conv2d(6, kernel_size=3))
        self.genome.add(Conv2d(8, kernel_size=3))
        self.genome.add(Conv2d(5, kernel_size=3))
        self.evaluate_model([5, 1, 5, 5])

    def test_2d_after_linear(self):
        self.phenotype.output_size = (1, 32, 32)
        self.genome.linear_at_end = False
        self.genome.add(Linear(32*32))
        self.genome.add(Deconv2d(1))
        self.genome.add(Linear(32*32*3))
        self.genome.add(Deconv2d(4))
        self.assertEqual([Linear, Linear, Deconv2d, Deconv2d], [gene.__class__ for gene in self.genome.genes])
        self.evaluate_model([5, 1, 32, 32])
        x = Variable(torch.randn(5, 32*32))
        self.phenotype.create_model(x)
        self.train_step(self.phenotype, x)

    def test_linear_at_end_true(self):
        self.genome.linear_at_end = True
        self.genome.random = False
        self.genome.add(Conv2d(1))
        self.genome.add(Conv2d(3))
        self.genome.add(Conv2d(6))
        self.genome.add(Linear(activation_type="Sigmoid"))
        self.assertEqual([Conv2d, Conv2d, Conv2d, Linear], [gene.__class__ for gene in self.genome.genes])
        self.assertEqual([1, 3, 6], [self.genome.genes[i].out_channels for i in range(3)])

    def test_linear_at_end_false(self):
        self.genome.linear_at_end = False
        self.genome.random = False
        self.genome.add(Conv2d(1))
        self.genome.add(Conv2d(3))
        self.genome.add(Linear(activation_type="Sigmoid"))
        self.genome.add(Conv2d(6))
        self.assertEqual([Linear, Conv2d, Conv2d, Conv2d], [gene.__class__ for gene in self.genome.genes])
        self.assertEqual([1, 3, 6], [self.genome.genes[i].out_channels for i in range(1, 4)])

    def test_linear_at_end(self):
        self.genome.linear_at_end = False
        self.genome.add(Linear(10))
        self.genome.add(Linear(activation_type="Sigmoid"))
        self.assertEqual([Linear, Linear], [gene.__class__ for gene in self.genome.genes])
        self.evaluate_model([5, 1, 12, 12])

    def test_random_genome(self):
        self.genome.random = True
        self.genome.add(Linear(16))
        self.genome.add(Linear(32))
        self.genome.add(Linear(1), force_sequence=True)
        self.genome.add(Conv2d(3, kernel_size=3))
        self.genome.add(Conv2d(6, kernel_size=3))
        self.genome.add(Conv2d(9, kernel_size=3), force_sequence=True)
        self.assertEqual([Conv2d, Conv2d, Conv2d, Linear, Linear, Linear], [gene.__class__ for gene in self.genome.genes])
        self.assertSetEqual(set([3, 6, 9]), set([self.genome.genes[i].out_channels for i in range(3)]))
        self.assertSetEqual(set([16, 32, 1]), set([self.genome.genes[i].out_features for i in range(3, 6)]))
        self.evaluate_model([5, 3, 5, 5])

    def test_limit_layers(self):
        self.genome.max_layers = 2
        self.genome.add_layer_prob = 1
        self.genome.rm_layer_prob = 0
        for i in range(3):
            self.genome.mutate()
        self.assertEqual(self.genome.max_layers, len(self.genome.genes))

    def test_equal_genome_distance(self):
        self.genome.add(Linear(32))
        self.genome.add(Linear(16))
        other = copy.deepcopy(self.genome)
        self.assertEqual(0, self.genome.distance(other))

    def test_different_genome_distance(self):
        self.genome.add(Linear(32))
        self.genome.add(Linear(16))
        other = copy.deepcopy(self.genome)
        other.add(Linear(8))
        other.add(Linear(1))
        self.assertEqual(2, self.genome.distance(other))

    def test_equal_genome_distance_after_breed(self):
        self.genome.add(Linear(32))
        self.genome.add(Linear(16))
        self.genome.output_genes.append(Linear(activation_type='Sigmoid'))
        x = Variable(torch.randn(5, 64))
        self.phenotype.create_model(x)
        self.train_step(self.phenotype, x)
        phenotype2 = self.phenotype.breed(skip_mutation=True)
        self.assertEqual(0, self.genome.distance(phenotype2.genome))

    def test_crossover(self):
        # create and train the first genotype
        x = Variable(torch.randn(5, 32*32)).view(5, 1, 32, 32)
        self.genome.crossover_rate = 1
        self.genome.add(Conv2d(16))
        self.genome.add(Linear(32))
        self.genome.add(Linear(16))
        self.genome.output_genes.append(Linear(activation_type='Sigmoid'))
        # create the mate genotype
        mate = Genome()
        mate.add(Conv2d(4))
        mate.add(Conv2d(8))
        mate.add(Linear(activation_type="ReLU"))
        mate.add(Linear(16))
        # apply crossover
        self.genome.crossover(mate)
        # check if layers correspond to expected
        self.assertEqual([Conv2d, Conv2d, Linear, Linear], [gene.__class__ for gene in self.genome.genes])
        # evaluate the created model
        self.evaluate_model([5, 1, 32, 32])
        self.phenotype.create_model(x)
        self.train_step(self.phenotype, x)

    def test_crossover_phenotype(self):
        # create and train the first genotype
        x = Variable(torch.randn(5, 32*32)).view(5, 1, 32, 32)
        self.genome.crossover_rate = 1
        self.genome.add(Conv2d(16))
        self.genome.add(Linear(32))
        self.genome.output_genes.append(Linear(activation_type='Sigmoid'))
        self.phenotype.create_model(x)
        self.train_step(self.phenotype, x)
        # create the mate genotype
        mate = Genome()
        mate.add(Conv2d(8))
        mate.add(Linear(16))
        phenotype_mate = Phenotype(1, mate)
        mate.output_genes.append(Linear(activation_type='Sigmoid'))
        phenotype_mate.create_model(x)
        self.train_step(phenotype_mate, x)
        # breed with crossover
        child = self.phenotype.breed(skip_mutation=True, mate=phenotype_mate)
        # verify if the weights was copied
        self.assertTrue(mate.genes[0].module.weight.equal(child.genome.genes[0].module.weight))
        old_state = phenotype_mate.optimizer.state[phenotype_mate.optimizer.param_groups[0]['params'][0]]
        new_state = child.optimizer.state[child.optimizer.param_groups[0]['params'][0]]
        self.assertTrue(old_state['exp_avg'].equal(new_state['exp_avg']))

    def test_crossover_empty(self):
        # create and train the first genotype
        x = Variable(torch.randn(5, 32*32)).view(5, 1, 32, 32)
        self.genome.add(Conv2d(8))
        self.genome.output_genes.append(Linear(activation_type='Sigmoid'))
        # create the mate genotype
        mate = Genome(crossover_rate=1)
        mate.add(Linear(16))
        # apply crossover
        mate.crossover(self.genome)
        # check if layers correspond to expected
        self.assertEqual([Conv2d, Linear], [gene.__class__ for gene in mate.genes])
        # evaluate the created model
        self.evaluate_model([5, 1, 32, 32])
        self.phenotype.create_model(x)
        self.train_step(self.phenotype, x)

    def test_convert_phenotype_to_json(self):
        self.genome.add(Conv2d(4, activation_type="ReLU"))
        self.genome.add(Linear(32))
        self.genome.add(Linear(16))
        self.evaluate_model([5, 1, 8, 8])
        layers = json.loads(self.phenotype.to_json())
        self.assertEqual("ReLU", layers[0]["activation_type"])
        self.assertEqual(3, len(layers))
        self.assertEqual(["Conv2d", "Linear", "Linear"], [l["type"] for l in layers])

    def test_invalid_graph(self):
        self.phenotype.output_size = (1, 28, 28)
        self.genome.linear_at_end = False
        self.genome.add(Linear(1568))
        self.genome.add(Deconv2d(32))
        self.genome.add(Deconv2d(32))
        self.genome.add(Deconv2d(32))
        self.genome.add(Deconv2d(32))
        self.genome.add(Deconv2d(32))
        self.genome.output_genes.append(Deconv2d(1))
        x = Variable(torch.randn(5, 100)).view(5, 1, 10, 10)
        self.assertRaises(Exception, self.phenotype.create_model, x)

    def test_multiple_deconv2d(self):
        self.phenotype.output_size = (1, 28, 28)
        self.genome.linear_at_end = False
        self.genome.add(Linear(1568))
        self.genome.add(Deconv2d(32, kernel_size=3))
        self.genome.add(Deconv2d(32, kernel_size=3))
        self.genome.add(Deconv2d(32, kernel_size=3))
        self.genome.output_genes.append(Deconv2d(1))
        x = Variable(torch.randn(5, 100)).view(5, 1, 10, 10)
        self.phenotype.create_model(x)
        out = self.phenotype.model(x)
        self.assertEqual([28, 28], list(out.size()[2:]))

    def test_simple_deconv2d(self):
        self.phenotype.output_size = (1, 28, 28)
        self.genome.linear_at_end = False
        self.genome.add(Deconv2d(32))
        self.genome.output_genes.append(Deconv2d(1))
        x = Variable(torch.randn(5, 100)).view(5, 1, 10, 10)
        self.phenotype.create_model(x)
        out = self.phenotype.model(x)
        self.assertEqual([28, 28], list(out.size()[2:]))

    def test_simple_deconv2d_32(self):
        self.phenotype.output_size = (3, 32, 32)
        self.genome.linear_at_end = False
        self.genome.add(Deconv2d(32))
        self.genome.output_genes.append(Deconv2d(3))
        x = Variable(torch.randn(5, 100)).view(5, 1, 10, 10)
        self.phenotype.create_model(x)
        out = self.phenotype.model(x)
        self.assertEqual([32, 32], list(out.size()[2:]))

    def test_simple_deconv2d_64(self):
        self.phenotype.output_size = (3, 64, 64)
        self.genome.linear_at_end = False
        self.genome.add(Deconv2d(32))
        self.genome.output_genes.append(Deconv2d(3))
        x = Variable(torch.randn(5, 100)).view(5, 1, 10, 10)
        self.phenotype.create_model(x)
        out = self.phenotype.model(x)
        self.assertEqual([64, 64], list(out.size()[2:]))


if __name__ == '__main__':
    unittest.main()
