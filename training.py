"""
Runs a model on a single node across N-gpus.
"""
import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchnlp.random import set_seed

from classifier import Classifier


def main(hparams) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    set_seed(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------
    model = Classifier(hparams)

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=False,
        mode=hparams.metric_mode,
    )

    # ------------------------
    # 3 INIT LOGGERS
    # ------------------------
    # Tensorboard Callback
    tb_logger = TensorBoardLogger(
        save_dir=hparams.save_dir,
        version="version_" + datetime.now().strftime("%d-%n-%Y--%H-%M-%S"),
        name="",
        default_hp_metric=True
    )

    # Model Checkpoint Callback
    ckpt_path = Path(hparams.save_dir) / str(tb_logger.version) / "checkpoints"

    # --------------------------------
    # 4 INIT MODEL CHECKPOINT CALLBACK
    # -------------------------------
    best_metric_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        every_n_epochs=1,
        mode=hparams.metric_mode,
        save_weights_only=True
    )

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=True,
        callbacks=[best_metric_checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        gpus=hparams.gpus,
        log_gpu_memory="all",
        deterministic=True,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        log_every_n_steps=hparams.log_every_n_steps,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        max_time=hparams.max_time,
        val_check_interval=hparams.val_check_interval,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        limit_test_batches=hparams.limit_test_batches,
        # distributed_backend="dp",
    )
    # ------------------------
    # 6 START TRAINING
    # ------------------------
    trainer.fit(model, model.data)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to a config.yaml file")
    config_path = parser.parse_args().config

    with open(config_path, "r") as stream:
        try:
            hparams = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    hparams = Namespace(**hparams)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
