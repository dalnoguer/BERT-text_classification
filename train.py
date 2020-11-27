import os
import click
import yaml

from model import TextClassificationModel

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


@click.command()
@click.argument('experiment_path', default=".", type=click.Path(exists=True))
def train(experiment_path):
    assert os.path.exists(
        os.path.join(experiment_path, "experiment.yaml")
    ), "No experiment configuration was found, please create an experiment.yaml"

    with open(os.path.join(experiment_path, "experiment.yaml"),
              mode="r") as config_file:
        configuration_dict = yaml.load(config_file, Loader=yaml.Loader)

    model = TextClassificationModel(configuration_dict)
    tb_logger = pl_loggers.TensorBoardLogger('logs/')

    assert not os.path.exists(configuration_dict['Training']
                              ['snapshot_dir']), "Experiment already exists."

    checkpoint = ModelCheckpoint(filepath=os.path.join(
        configuration_dict['Training']['snapshot_dir'],
        'best_model_{epoch:02d}-{val_loss:.2f}'),
                                 verbose=True,
                                 monitor='val_loss',
                                 mode='min')

    trainer = Trainer(logger=tb_logger,
                      checkpoint_callback=checkpoint,
                      max_epochs=configuration_dict['Training']['epochs'],
                      gpus=configuration_dict['Training']['gpus'])

    trainer.fit(model)

    trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path)


if __name__ == '__main__':
    train()
