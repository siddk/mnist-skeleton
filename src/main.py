"""
main.py

Core script for downloading and running MNIST Data, initializing and training various neural models, and logging
training statistics.
"""
from argparse import Namespace
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from tap import Tap

from models.cnn import CNN
from models.feedforward import FeedForward

import pytorch_lightning as pl


class ArgumentParser(Tap):
    # Weights & Biases Parameters
    run_name: str = None                  # Informative Run-ID for WandB
    project: str = 'mnist-skeleton'       # Project Name for WandB Logging
    data_dir: str = 'data/'               # Where to download data
    save_dir: str = 'checkpoints/'        # Where to save WandB Artifacts
    sync: bool = False                    # Boolean if developing (no WandB Logging!)

    # GPUs
    gpus: int = 0                         # Number of GPUs to run with

    # Model Parameters
    model: str = 'cnn'                    # Model type to run -- one of < feedforward | cnn >

    # FeedForward Network Parameters
    ff_1: int = 128                       # Number of neurons in first hidden layer
    ff_2: int = 256                       # Number of neurons in second hidden layer

    # CNN Parameters
    cnn_conv1: int = 10                   # Number of Channels for First Convolution Layer
    cnn_conv2: int = 20                   # Number of Channels for Second Convolution Layer
    kernel_size: int = 5                  # Kernel Size (patch size) for Convolution Layers
    cnn_ff: int = 50                      # Number of neurons in projection layer after Convolution Layers

    # Training Parameters
    bsz: int = 64                         # Batch Size
    opt: str = 'adam'                     # Optimizer to use -- one of < adam | sgd >
    lr: float = 0.001                     # Learning Rate


def main():
    # Parse Arguments --> Convert from Namespace --> Dict --> Namespace because of weird WandB Bug
    args = Namespace(**ArgumentParser().parse_args().as_dict())

    # Create Logger
    if args.run_name is None:
        run_name = "%s-%s-%d-%.1g" % (args.model, args.opt, args.bsz, args.lr) + '+' + \
                   datetime.now().strftime('%m/%d-[%H:%M]')
    else:
        run_name = args.run_name + '+' + datetime.now().strftime('%m/%d-[%H:%M]')
    wandb = WandbLogger(name=run_name, save_dir=args.save_dir, project=args.project, offline=not args.sync)

    # Create MNIST Module
    if args.model == 'feedforward':
        nn = FeedForward(args)
    elif args.model == 'cnn':
        nn = CNN(args)

    # Prepare Data and Populate Data Loader
    nn.prepare_data()
    nn.train_dataloader()

    # Create Trainer
    trainer = pl.Trainer(default_save_path=args.save_dir, max_epochs=10, logger=wandb, gpus=args.gpus)

    # Watch Histogram of Gradients
    wandb.experiment.watch(nn, log='gradients', log_freq=100)

    # Fit
    trainer.fit(nn)


if __name__ == "__main__":
    main()
