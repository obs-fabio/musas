"""
Trainer.

Provides utilities to train classifiers over all folds of a DataModule,
while skipping folds that have already been trained.
"""

import os
import argparse

import lps_ml.core as ml_core
import lps_ml.utils.default as ml_default
import lps_ml.model.cnn as lps_cnn


class Trainer:
    """
    Runner for classifier cross-validation training.

    Each CV fold is trained independently and stored in:

        <output_dir>/fold_00/
        <output_dir>/fold_01/
        ...
    """

    def __init__(self, args: argparse.Namespace, output_dir: str):
        self.args = args
        self.base_dir = output_dir

        os.makedirs(self.base_dir, exist_ok=True)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, default_output_dir: str) -> None:
        """
        Add classifier model and training arguments.
        """
        ml_default.add_training_args(parser, default_output_dir=default_output_dir)
        lps_cnn.CNN2D.add_args(parser)

    def get_fold_dir(self, fold: int) -> str:
        """Return the output directory for a CV fold."""
        return os.path.join(self.base_dir, f"fold_{fold:02d}")

    def train_fold(
        self,
        dm: ml_core.BaseDataModule,
        fold: int
    ) -> str:
        """
        Train one CV fold.
        """

        fold_dir = self.get_fold_dir(fold)

        trainer_args = argparse.Namespace(**vars(self.args))
        trainer_args.output_dir = fold_dir
        trainer, checkpoint = ml_default.trainer_from_args(trainer_args)

        best_checkpoint = checkpoint.get_best()

        if os.path.isfile(best_checkpoint):
            print(
                f"Fold {fold:02d} already trained. "
                f"Skipping: {best_checkpoint}"
            )
            return best_checkpoint

        os.makedirs(fold_dir, exist_ok=True)

        dm.set_fold(fold)

        model = lps_cnn.CNN2D.from_args(self.args, dm)

        print()
        print(f"Training fold {fold:02d}")
        print(f"Output: {fold_dir}")
        print()

        trainer.fit(model, datamodule=dm)

        return checkpoint.get_best()

    def train(self, dm: ml_core.BaseDataModule) -> dict[int, str]:
        """
        Train all CV folds of a DataModule.
        """

        dm.setup()

        n_folds = dm.get_n_folds()

        print()
        print(f"Training {n_folds} folds")
        print()

        checkpoints = {}

        for fold in range(n_folds):

            checkpoint = self.train_fold(
                dm=dm,
                fold=fold
            )

            checkpoints[fold] = checkpoint

        return checkpoints
