"""
Run Synthetic Experiments and save the dataset info tables to CSV files.

tensorboard --logdir . --port 8888 --bind_all
"""
import os
import argparse

import torch
import lps_ml.utils.general as ml_utils

import acoustic_simulator.synthetic_experiments as as_exp
import acoustic_simulator.trainer as as_trn
import acoustic_simulator.evaluator as as_eval

def _main():
    """Main function for the dataset info tables."""

    parser = argparse.ArgumentParser(description="Train an MLP classifier on iara.")
    as_trn.Trainer.add_args(parser, default_output_dir="./results/synthectic_experiments")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    for exp in as_exp.SyntheticExperiment:

        exp_dir = os.path.join(output_dir, exp.as_str())

        datamodules = exp.build_datamodules()

        trainers = {}
        checkpoints = {}

        for exp_name, dm in datamodules.items():

            exp_sub_dir = os.path.join(exp_dir, exp_name)

            print(ml_utils.format_header(60, str(exp)))
            print(ml_utils.format_header(40, exp_name))

            print(dm.to_compile_df())

            trainer = as_trn.Trainer(args, exp_sub_dir)
            fold_checkpoints = trainer.train(dm)

            trainers[exp_name] = trainer
            checkpoints[exp_name] = fold_checkpoints

        for exp_name, dm in datamodules.items():

            evaluator = as_eval.Evaluator(
                trainer=trainers[exp_name],
                checkpoints=checkpoints[exp_name],
            )

            evaluator.evaluate(dm)

            evaluator.cross_evaluate(
                source_name=exp_name,
                target_dms=datamodules,
            )

        as_eval.Evaluator.compile_experiment_results(
            exp_dir=exp_dir,
            sub_experiments=list(datamodules.keys()),
        )


if __name__ == "__main__":
    _main()
