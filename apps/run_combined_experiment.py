"""
Run Combined Domain Experiment.

Train each fold using the combined IARA + IEMANJA dataset
and evaluate each validation fold separately by source module.
"""

import argparse

import torch
import lps_ml.utils.general as ml_utils

import acoustic_simulator.domain_experiment as as_exp
import acoustic_simulator.trainer as as_trn
import acoustic_simulator.evaluator as as_eval


def _main():

    parser = argparse.ArgumentParser(description="Run combined domain experiment.")
    as_trn.Trainer.add_args(parser, default_output_dir="./results/combined_experiment")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    exp = as_exp.DomainExperiment()

    dm = exp.build_combined_datamodule()
    trainer = as_trn.Trainer(args, args.output_dir)
    checkpoints = trainer.train(dm)

    evaluator = as_eval.Evaluator(trainer=trainer, checkpoints=checkpoints)
    evaluator.evaluate(dm)
    evaluator.evaluate_by_module(dm)

if __name__ == "__main__":
    _main()
