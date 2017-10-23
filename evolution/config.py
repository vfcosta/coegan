from munch import Munch


params = {
    "evolution": {
        "max_generations": 50,
        "elitism": 0.0,
        "tournament_size": 2,
        "max_layers": 5,
        "sequential_layers": False,  # when true, genes/layers will be added sequentially
        "add_layer_prob": 0.20,
        "rm_layer_prob": 0.10,
        "gene_mutation_prob": 0.10,
        "crossover_rate": 0.0,
        "conv_beginning": False,
        "freeze_when_change": False,
        "evaluation": {
            "type": "all-vs-all",  # all-vs-all | random | all-vs-best | all-vs-kbest | all-vs-species-best
            "best_size": 1,
            "iterations": 1,
            "initialize_all": True  # apply all-vs-all for the first time type is all-vs-best, all-vs-kbest or all-vs-species-best
        },
        "speciation": {
            "size": 3,
            "keep_best": True,
            "threshold": 1,
        },
        "fitness": {
            "discriminator": "loss",  # AUC | loss | random
            "generator": "FID",  # AUC | FID | loss | random
            "fid_sample_size": 1000,  # TODO: this is also the sample size for the inception score
            "fid_dimension": 2048  # 2048 | 768 | 192 | 64
        }
    },
    "gan": {
        "dataset": "MNIST",  # CelebA64Cropped | CelebA32Cropped | MNIST | FashionMNIST | CIFAR10
        "dataset_classes": None,  # select a subset of classes to use (set to None to use all)
        "batches_limit": 20,
        "batch_size": 64,
        "critic_iterations": 1,
        "type": "gan",  # gan | wgan | lsgan | rsgan | rasgan
        "label_smoothing": False,
        "batch_normalization": False,
        "pixelwise_normalization": False,
        "use_wscale": False,
        "use_minibatch_stddev": False,
        "dropout": False,
        "discriminator": {
            "population_size": 10,
            "simple_layers": False,
            "fixed": False,
            "use_gradient_penalty": False,
            "gradient_penalty_lambda": 10,
        },
        "generator": {
            "population_size": 10,
            "simple_layers": False,
            "fixed": False,
        },
    },
    "optimizer": {
        "type": "Adam",  # Adam | RMSprop | SGD | Adadelta
        "copy_optimizer_state": False,
        "learning_rate": 1e-3,
        "weight_decay": 0,
    },
    "layer": {
        "keep_weights": True,
        "resize_weights": True,
        "conv2d": {
            "max_channels_power": 7,  # max number of channels is 2**max_channels_power
            "random_out_channels": True  # when false, will calculate output_channels based on in_channels
        }
    },
    "stats": {
        "num_generated_samples": 36,
        "print_interval": 1,
        "print_best_amount": 1,
        "display_validation_stats": False,
        "calc_inception_score": True,
        "calc_fid_score": True,
        "calc_rmse_score": True,
        "save_best_model": True,
        "save_best_interval": 5,
        "notify": True,
        "min_notification_interval": 30
    }
}
config = Munch.fromDict(params)
