"""
Run Domain Experiment dataset information.

tensorboard --logdir . --port 8888 --bind_all
"""
import os
import argparse

import torch
import lps_ml.utils.general as ml_utils

import acoustic_simulator.domain_experiment as as_exp
import acoustic_simulator.trainer as as_trn
import acoustic_simulator.evaluator as as_eval


def _print_fold_volume(name, dm):
    """Print the number of samples in the first fold."""

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_total = n_train + n_val

    print(ml_utils.format_header(40, f"{name} - Fold 0 Volume"))

    print(f"Train samples: {n_train:,}")
    print(f"Val samples:   {n_val:,}")
    print(f"Total samples: {n_total:,}")

def _print_dataset_info(name, dm):
    """Print dataset information and first-fold volume."""

    print(ml_utils.format_header(60, f"{name} Dataset Info"))

    print(dm.to_compile_df())

    _print_fold_volume(name, dm)

    print(ml_utils.format_header(40))
    print(dm.to_df())

def _main():
    """Main function for the dataset info tables."""

    parser = argparse.ArgumentParser(description="Run domain transfer experiment.")
    as_trn.Trainer.add_args(parser, default_output_dir="./results/domain_experiment")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    exp = as_exp.DomainExperiment()

    datamodules = exp.build_domain_datamodules()

    trainers = {}
    checkpoints = {}

    for name, dm in datamodules.items():

        output_dir = os.path.join(args.output_dir, name)
        trainer = as_trn.Trainer(args, output_dir)

        checkpoints[name] = trainer.train(dm)
        trainers[name] = trainer

    for source_name, dm in datamodules.items():

        evaluator = as_eval.Evaluator(
            trainer=trainers[source_name],
            checkpoints=checkpoints[source_name],
        )

        evaluator.evaluate(dm)

        evaluator.cross_evaluate(
            source_name=source_name,
            target_dms=datamodules,
        )

    as_eval.Evaluator.compile_experiment_results(
        exp_dir=args.output_dir,
        sub_experiments=list(datamodules.keys()),
    )

if __name__ == "__main__":
    _main()
