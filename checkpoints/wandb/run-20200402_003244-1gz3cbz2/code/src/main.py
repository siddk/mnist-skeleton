"""
main.py

Core script for downloading and running MNIST Data, initializing and training various neural models, and logging
training statistics.
"""
from argparse import Namespace
from pytorch_lightning.loggers import WandbLogger
from tap import Tap

from models.feedforward import FeedForward

import pytorch_lightning as pl


class ArgumentParser(Tap):
    # Weights & Biases Parameters
    run_name: str                         # Informative Run-ID for WandB
    project: str = 'mnist-skeleton'       # Project Name for WandB Logging
    data_dir: str = 'data/'               # Where to download data
    save_dir: str = 'checkpoints/'        # Where to save WandB Artifacts

    # GPUs
    gpus: int = 0                         # Number of GPUs to run with

    # Model Parameters
    model: str = 'feedforward'            # Model type to run -- one of < feedforward | cnn >

    # FeedForward Network Parameters
    ff_1: int = 128                       # Number of neurons in first hidden layer
    ff_2: int = 256                       # Number of neurons in second hidden layer

    # Training Parameters
    bsz: int = 64                         # Batch Size
    opt: str = 'adam'                     # Optimizer to use -- one of < adam | sgd >
    lr: float = 0.01                      # Learning Rate


def main():
    # Parse Arguments --> Convert from Namespace --> Dict --> Namespace because of weird WandB Bug
    args = Namespace(**ArgumentParser().parse_args().as_dict())

    # Create Logger
    wandb = WandbLogger(name=args.run_name, save_dir=args.save_dir, project=args.project)

    # Create MNIST Module
    if args.model == 'feedforward':
        nn = FeedForward(args)

    # Prepare Data and Populate Data Loader
    nn.prepare_data()
    nn.train_dataloader()

    # Create Trainer
    trainer = pl.Trainer(default_save_path=args.save_dir, max_epochs=10, logger=wandb)

    # Fit
    trainer.fit(nn)


if __name__ == "__main__":
    main()
