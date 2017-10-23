import torch
from torch.autograd import Variable
import unittest
from evolution.discriminator import Discriminator
from evolution.generator import Generator
from util import tools
import os
import shutil


class TestGAN(unittest.TestCase):

    def setUp(self):
        self.test_path = "/tmp/egan"
        os.makedirs(self.test_path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_path, ignore_errors=True)

    def assert_state_dict_equal(self, s1, s2):
        self.assertEqual(len(s1), len(s2))
        for k, v in s1.items():
            self.assertTrue(v.equal(s2[k]))

    def test_serialization(self):
        images = tools.cuda(Variable(torch.randn(5, 100)).view(5, 1, 10, 10))
        input_shape = images[0].size()
        discriminator = Discriminator(output_size=1, input_shape=[1]+list(input_shape))
        discriminator.setup()
        generator = Generator(output_size=input_shape)
        generator.setup()

        generator = tools.cuda(generator)
        discriminator = tools.cuda(discriminator)
        discriminator.do_train(generator, images)
        generator.do_train(discriminator, images)

        # save and load the discriminator
        discriminator_path = f"{self.test_path}/discriminator.pkl"
        discriminator.save(discriminator_path)
        loaded_discriminator = Discriminator.load(discriminator_path)
        self.assert_state_dict_equal(discriminator.state_dict(), loaded_discriminator.state_dict())

        # save and load the generator
        generator_path = f"{self.test_path}/generator.pkl"
        generator.save(generator_path)
        loaded_generator = Generator.load(generator_path)
        loaded_generator = tools.cuda(loaded_generator)
        generator = tools.cuda(generator)
        self.assert_state_dict_equal(generator.state_dict(), loaded_generator.state_dict())
        # check if the loaded generator will generate images in the same way that the original generator
        noise = generator.generate_noise(1, volatile=True)
        diff = generator(noise) - loaded_generator(noise)
        self.assertAlmostEqual(0, diff.sum().item(), 6)
        # execute a train step and now it should be different
        generator.do_train(tools.cuda(discriminator), images)
        self.assertFalse(generator(noise).equal(loaded_generator(noise)))
