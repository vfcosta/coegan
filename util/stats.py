# -*- coding: future_fstrings -*-
import torch
import torchvision.utils as vutils
import pandas as pd
from torch.autograd import Variable
import numpy as np
import logging
from tensorboardX import SummaryWriter
from evolution.config import config
import os
from util import tools
import shutil
from util.notifier import notify
from datetime import datetime
from metrics import rmse_score

logger = logging.getLogger(__name__)


class Stats:

    def __init__(self, log_dir=None):
        self.test_noise = None
        self.writer = SummaryWriter(log_dir=log_dir)
        self.input_shape = None
        self.last_notification = None
        os.makedirs("%s/images" % self.writer.file_writer.get_logdir())
        shutil.copy("./evolution/config.py", self.writer.file_writer.get_logdir())  # copy setup file into log dir

    def save_data(self, epoch, g_pop, d_pop, save_best_model=False):
        if not os.path.isfile(os.path.join(self.writer.file_writer.get_logdir(), "./generator_noise.pt")):
            shutil.copy("./generator_noise.pt", self.writer.file_writer.get_logdir())  # copy noise file into log dir

        epoch_dir = f"{self.writer.file_writer.get_logdir()}/generations/{epoch:03d}"
        os.makedirs(epoch_dir)
        global_data_values = {}
        for name, pop in [("d", d_pop), ("g", g_pop)]:
            phenotypes = pop.phenotypes()
            global_data_values[f"species_{name}"] = len(pop.species_list)
            global_data_values[f"speciation_threshold_{name}"] = pop.speciation_threshold
            global_data_values[f"invalid_{name}"] = sum([p.invalid for p in phenotypes])
            # generate data for current generation
            columns = ["loss", "trained_samples", "layers", "genes_used",
                       "model", "species_index", "fitness", "generation", "age"]
            if name == "g":
                columns.append("fid_score")
                columns.append("inception_score")
                columns.append("rmse_score")
            df = pd.DataFrame(index=np.arange(0, len(phenotypes)), columns=columns)
            j = 0
            for i, species in enumerate(pop.species_list):
                for p in species:
                    values = [p.error, p.trained_samples, len(p.genome.genes), np.mean([g.used for g in p.genome.genes]),
                              p.to_json(), i, p.fitness(), p.genome.generation, p.genome.age]
                    if name == "g":
                        values.append(p.fid_score)
                        values.append(p.inception_score_mean)
                        values.append(p.rmse_score)
                    df.loc[j] = values
                    j += 1
            df.sort_values('fitness').reset_index(drop=True).to_csv(f"{epoch_dir}/data_{name}.csv")

        # generate image for each G
        os.makedirs(f"{epoch_dir}/images")
        for i, g in enumerate(g_pop.sorted()):
            if not g.valid():
                continue
            self.generate_image(g, path=f"{epoch_dir}/images/generator-{i:03d}.png")
            if i == 0 and save_best_model:
                g.save(f"{epoch_dir}/generator.pkl")

        if save_best_model:
            d_pop.sorted()[0].save(f"{epoch_dir}/discriminator.pkl")

        # append values into global data
        global_data = pd.DataFrame(data=global_data_values, index=[epoch])
        with open(f"{self.writer.file_writer.get_logdir()}/generations/data.csv", 'a') as f:
            global_data.to_csv(f, header=epoch == 0)

    def generate_image(self, G, path=None):
        if not G.valid():
            return None
        test_images = G(self.test_noise).detach()
        grid_images = [torch.from_numpy((test_images[k, :].data.cpu().numpy().reshape(self.input_shape) + 1)/2)
                        for k in range(config.stats.num_generated_samples)]
        grid = vutils.make_grid(grid_images, normalize=False, nrow=int(config.stats.num_generated_samples**(1/2)))
        # store grid images in the run folder
        if path is not None:
            vutils.save_image(grid, path)
        return grid

    def generate(self, input_shape, g_pop, d_pop, epoch, num_epochs, train_loader, validation_loader):
        if epoch % config.stats.print_interval != 0 and epoch != num_epochs - 1:
            return

        generators = g_pop.sorted()
        discriminators = d_pop.sorted()
        G = g_pop.best()
        D = d_pop.best()
        G.eval()
        D.eval()

        # this should never ocurr!
        if G.invalid or D.invalid:
            logger.error("invalid D or G")
            return

        self.input_shape = input_shape
        if self.test_noise is None:
            self.test_noise = G.generate_noise(config.stats.num_generated_samples, volatile=True).cpu()
            # display noise only once
            # grid_noise = vutils.make_grid(self.test_noise.data, normalize=True, scale_each=True, nrow=4)
            # self.writer.add_image('Image/Noise', grid_noise)

        if config.stats.calc_rmse_score:
            rmse_score.initialize(train_loader, config.evolution.fitness.fid_sample_size)
            for g in generators:
                g.calc_rmse_score()

        if config.stats.calc_inception_score:
            for g in generators:
                g.inception_score()
            self.writer.add_scalars('Training/Inception_score', {"Best_G": G.inception_score_mean}, epoch)
        if G.fid_score is not None:
            self.writer.add_scalars('Training/Fid_score', {"Best_G": G.fid_score}, epoch)

        self.save_data(epoch, g_pop, d_pop, config.stats.save_best_model and (epoch == num_epochs-1 or epoch % config.stats.save_best_interval == 0))

        self.writer.add_scalars('Training/Trained_samples', {"Best_D": D.trained_samples,
                                                             "Best_G": G.trained_samples,
                                                             "D": sum([p.trained_samples for p in discriminators])/len(discriminators),
                                                             "G": sum([p.trained_samples for p in generators])/len(generators)
                                                             }, epoch)
        self.writer.add_scalars('Training/Loss', {"Best_D": D.error, "Best_G": G.error}, epoch)
        self.writer.add_scalars('Training/Fitness', {"Best_D": D.fitness(), "Best_G": G.fitness()}, epoch)
        self.writer.add_scalars('Training/Generation', {"Best_D": D.genome.generation, "Best_G": G.genome.generation}, epoch)
        self.writer.add_histogram('Training/Loss/D', np.array([p.error for p in discriminators]), epoch)
        self.writer.add_histogram('Training/Loss/G', np.array([p.error for p in generators]), epoch)
        self.writer.add_histogram('Training/Trained_samples/D', np.array([p.trained_samples for p in discriminators]), epoch)
        self.writer.add_histogram('Training/Trained_samples/G', np.array([p.trained_samples for p in generators]), epoch)

        # generate images with the best perfomings G's
        for i, gen in enumerate(generators[:config.stats.print_best_amount]):
            image_path = None
            if i == 0:
                image_path = '%s/images/generated-%05d.png' % (self.writer.file_writer.get_logdir(), epoch)
            grid = self.generate_image(gen, path=image_path)
            self.writer.add_image('Image/Best_G/%d' % i, grid, epoch)

        # write architectures for best G and D
        self.writer.add_text('Graph/Best_G', str([str(p) for p in generators[:config.stats.print_best_amount]]), epoch)
        self.writer.add_text('Graph/Best_D', str([str(p) for p in discriminators[:config.stats.print_best_amount]]), epoch)

        # apply best G and D in the validator dataset
        # FIXME: the validation dataset was already evaluated at this point. Just reuse the data.
        if config.stats.display_validation_stats:
            d_errors_real, d_errors_fake, g_errors = [], [], []
            for n, (images, _) in enumerate(validation_loader):
                images = tools.cuda(Variable(images))
                batch_size = images.size(0)

                d_errors_real.append(D.step_real(images))
                fake_error, _ = D.step_fake(G, batch_size)
                d_errors_fake.append(fake_error)
                g_errors.append(G.step(D, batch_size).data[0])

            # display validation metrics
            self.writer.add_scalars('Validation/D/Loss', {'Real': np.mean(d_errors_real),
                                                          'Fake': np.mean(d_errors_fake)}, epoch)
            self.writer.add_scalars('Validation/Loss', {'Best_D': np.mean(d_errors_real + d_errors_fake),
                                                        'Best_G': np.mean(g_errors)}, epoch)

        # display architecture metrics
        self.writer.add_scalars('Architecture/Layers', {'Best_D': len(D.genome.genes),
                                                        'Best_G': len(G.genome.genes),
                                                        'D': np.mean([len(p.genome.genes) for p in discriminators]),
                                                        'G': np.mean([len(p.genome.genes) for p in generators])
                                                        }, epoch)
        self.writer.add_histogram('Architecture/Layers/D', np.array([len(p.genome.genes) for p in discriminators]), epoch)
        self.writer.add_histogram('Architecture/Layers/G', np.array([len(p.genome.genes) for p in generators]), epoch)

        self.writer.add_scalars('Architecture/Invalid', {'D': sum([p.invalid for p in discriminators]),
                                                         'G': sum([p.invalid for p in generators])
                                                         }, epoch)

        self.writer.add_scalars('Architecture/Species', {"D": len(d_pop.species_list), "G": len(g_pop.species_list)}, epoch)
        self.writer.add_scalars('Architecture/Speciation_Threshold', {"D": int(d_pop.speciation_threshold),
                                                                      "G": int(g_pop.speciation_threshold)}, epoch)

        best_d_used = np.mean([g.used for g in D.genome.genes])
        best_g_used = np.mean([g.used for g in G.genome.genes])
        d_used = np.mean([np.mean([g.used for g in p.genome.genes]) for p in discriminators])
        g_used = np.mean([np.mean([g.used for g in p.genome.genes]) for p in generators])
        self.writer.add_scalars('Architecture/Genes_reuse', {'Best_D': best_d_used, 'Best_G': best_g_used,
                                                             'D': d_used, 'G': g_used}, epoch)

        logger.debug("\n%s: D: %s G: %s", epoch, D.error, G.error)
        logger.debug(G); logger.debug(G.model)
        logger.debug(D); logger.debug(D.model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug((f"memory_allocated: {torch.cuda.memory_allocated()}, "
                          f"max_memory_allocated: {torch.cuda.max_memory_allocated()}, "
                          f"memory_cached: {torch.cuda.memory_cached()}, "
                          f"max_memory_cached: {torch.cuda.max_memory_cached()}"))

        if config.stats.notify and \
                (self.last_notification is None or
                 (datetime.now() - self.last_notification).seconds//60 > config.stats.min_notification_interval):
            self.last_notification = datetime.now()
            notify(f"Epoch {epoch}: G {G.fitness():.2f}, D: {D.error:.2f}")

        # graph plotting
        # dummy_input = Variable(torch.randn(28, 28)).cuda()
        # self.writer.add_graph(D.model, (dummy_input, ))
        # dummy_input = Variable(torch.randn(10, 10)).cuda()
        # self.writer.add_graph(G.model, (dummy_input, ))

        # flush writer to avoid memory issues
        self.writer.scalar_dict = {}
