"""
Classifier evaluation utilities.
"""

import os

import pandas as pd
import numpy as np
import torch
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import seaborn as sns

import lps_ml.core as ml_core
import lps_ml.model.cnn as lps_cnn
def _save_heatmap(values, formatted, filename):

    formatted_multiline = np.empty_like(formatted, dtype=object)

    for idx, value in np.ndenumerate(formatted):
        text = str(value)

        if "+-" in text:
            mean, std = text.split("+-")

            mean = mean.strip().replace("%", "")
            std = std.strip().replace("%", "")

            formatted_multiline[idx] = f"{mean}%\n± {std}%"
        else:
            formatted_multiline[idx] = text

    f = 0.65
    figsize = (7*f, 6*f)
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        values,
        annot=formatted_multiline,
        fmt="",
        ax=ax,
        square=True,
        linewidths=0.5,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annot_kws={"fontsize": 12, "linespacing": 1.15},
        cbar_kws={
            "format": lambda x, _: f"{x * 100:.0f}%",
            "shrink": 0.75
        },
    )

    ax.set_xlabel("Trained on", fontsize=16, labelpad=10)
    ax.set_ylabel("Evaluated on", fontsize=16, labelpad=10)

    ax.tick_params(axis="x", labelsize=13, rotation=0)
    ax.tick_params(axis="y", labelsize=13, rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(filename, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

class Evaluator:
    """
    Evaluate trained classifier models.

    Internal evaluation:

        <output_dir>/fold_XX/val_predictions.csv
        <output_dir>/metrics.csv

    Cross evaluation:

        <output_dir>/cross_eval/<target_sub_exp>.csv
        <output_dir>/cross_eval/metrics.csv
    """

    def __init__(
        self,
        trainer,
        checkpoints: dict[int, str],
    ):
        self.trainer = trainer
        self.args = trainer.args
        self.base_dir = trainer.base_dir
        self.checkpoints = checkpoints

    @staticmethod
    def calculate_classification_metrics(target, predict) -> dict[str, float]:

        return {
            "acc": float(
                sk_metrics.accuracy_score(target, predict)
            ),
            "recall": float(
                sk_metrics.recall_score(target, predict, average="macro", zero_division=0)
            ),
            "f1": float(
                sk_metrics.f1_score(target, predict, average="macro", zero_division=0)
            ),
        }

    def load_model(self, fold: int):
        """
        Load the model corresponding to a trained fold.
        """
        checkpoint_path = self.checkpoints[fold]
        model = lps_cnn.CNN2D.load_from_checkpoint(checkpoint_path)
        model.eval()
        return model

    @staticmethod
    def predict(model, dataloader) -> tuple[list, list, list]:

        targets = []
        predictions = []

        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():

            for data, target in dataloader:

                data = data.to(device)
                output = model(data)

                if output.ndim == 1:
                    predict = (output >= 0.5).long()
                elif output.ndim == 2 and output.shape[1] == 1:
                    predict = (output[:, 0] >= 0.5).long()
                else:
                    predict = torch.argmax(output, dim=1)

                targets.extend(target.cpu().numpy())
                predictions.extend(predict.cpu().numpy())

        ids = list(range(len(targets)))

        return (
            ids,
            targets,
            predictions,
        )

    def evaluate_fold(self, dm: ml_core.BaseDataModule, fold: int) -> dict[str, float]:

        fold_dir = os.path.join(self.base_dir, f"fold_{fold:02d}")
        filename = os.path.join(fold_dir, "val_predictions.csv")

        if os.path.isfile(filename):

            predictions_df = pd.read_csv(filename)

            print(
                f"Skipping evaluation of fold {fold:02d}: "
                f"{filename} already exists."
            )

            return self.calculate_classification_metrics(
                predictions_df["target"],
                predictions_df["predict"],
            )

        dm.set_fold(fold)

        model = self.load_model(fold)

        ids, targets, predictions = self.predict(model, dm.val_dataloader())

        predictions_df = pd.DataFrame({
            "id": ids,
            "target": targets,
            "predict": predictions,
        })

        predictions_df.to_csv(filename, index=False)

        return self.calculate_classification_metrics(targets, predictions)

    def evaluate(self, dm: ml_core.BaseDataModule) -> list[dict]:

        metrics_filename = os.path.join(self.base_dir, "metrics.csv")

        if os.path.isfile(metrics_filename):

            print(
                f"Skipping internal evaluation: "
                f"{metrics_filename} already exists."
            )

            return pd.read_csv(metrics_filename).to_dict("records")

        dm.setup()

        results = []

        for fold in self.checkpoints:

            metrics = self.evaluate_fold(dm=dm, fold=fold)

            results.append({
                "fold": fold,
                **metrics,
            })

        df = pd.DataFrame(results, columns=["fold", "acc", "recall", "f1"])
        df.to_csv(metrics_filename, index=False)
        return results

    def evaluate_fold_by_module(self, dm: ml_core.CombinedDataModule, fold: int) -> list[dict]:

        dm.set_fold(fold)
        model = self.load_model(fold)
        loaders = dm.get_dataloaders_by_module("val")

        fold_dir = os.path.join(self.base_dir,"module_eval",f"fold_{fold:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        results = []

        for module_name, dataloader in loaders.items():

            filename = os.path.join(fold_dir, f"{module_name}.csv")

            if os.path.isfile(filename):
                predictions_df = pd.read_csv(filename)

                targets = predictions_df["target"]
                predictions = predictions_df["predict"]

            else:

                ids, targets, predictions = self.predict(model, dataloader)

                predictions_df = pd.DataFrame({
                    "id": ids,
                    "target": targets,
                    "predict": predictions,
                })

                predictions_df.to_csv(filename, index=False)

            metrics = self.calculate_classification_metrics(targets, predictions)

            results.append({
                "module": module_name,
                "fold": fold,
                **metrics,
            })

        return results

    def evaluate_by_module(self, dm: ml_core.CombinedDataModule) -> list[dict]:

        dm.setup()

        results = []

        for fold in self.checkpoints:
            fold_results = self.evaluate_fold_by_module(dm=dm, fold=fold)
            results.extend(fold_results)

        metrics_df = pd.DataFrame(
            results,
            columns=[
                "module",
                "fold",
                "acc",
                "recall",
                "f1",
            ],
        )

        module_names = metrics_df["module"].unique()

        for module_name in module_names:

            module_df = metrics_df[metrics_df["module"] == module_name][
                ["fold", "acc", "recall", "f1"]
            ]

            filename = os.path.join(self.base_dir, f"{module_name}.csv")
            module_df.to_csv(filename, index=False)

        summary = []

        for module_name in module_names:

            module_df = metrics_df[metrics_df["module"] == module_name]
            row = {"module": module_name}

            for metric in ["acc", "recall", "f1"]:

                mean = module_df[metric].mean()
                std = module_df[metric].std()

                row[metric] = f"{mean * 100:.1f} +- {std * 100:.1f} %"

            summary.append(row)

        summary_df = pd.DataFrame(
            summary,
            columns=[
                "module",
                "acc",
                "recall",
                "f1",
            ],
        )

        metrics_filename = os.path.join(self.base_dir, "metrics.csv")
        summary_df.to_csv(metrics_filename, index=False)
        return results

    def cross_evaluate_target(
        self,
        target_dm: ml_core.BaseDataModule,
        target_name: str,
    ) -> list[dict]:

        cross_eval_dir = os.path.join(self.base_dir, "cross_eval")
        os.makedirs(cross_eval_dir, exist_ok=True)

        predictions_filename = os.path.join(cross_eval_dir, f"{target_name}.csv")

        if os.path.isfile(predictions_filename):

            print(
                f"Skipping cross evaluation of '{target_name}': "
                f"{predictions_filename} already exists."
            )

            predictions_df = pd.read_csv(predictions_filename)

            results = []

            for fold in self.checkpoints:

                column = f"fold_{fold:02d}_pred"

                metrics = self.calculate_classification_metrics(
                    predictions_df["target"],
                    predictions_df[column],
                )

                results.append({
                    "target_sub_exp": target_name,
                    "fold": fold,
                    **metrics,
                })

            return results

        target_dm.setup()
        target_dm.set_fold(0)

        predictions_by_fold = {}
        val_ids = None
        targets = None

        for fold in self.checkpoints:

            model = self.load_model(fold)

            current_ids, current_targets, predictions = self.predict(model,
                                                                     target_dm.all_dataloader())

            if val_ids is None:
                val_ids = current_ids
                targets = current_targets

            elif current_ids != val_ids:
                raise ValueError(
                    "Validation IDs differ between folds "
                    f"for target '{target_name}'."
                )

            predictions_by_fold[f"fold_{fold:02d}_pred"] = predictions

        predictions_df = pd.DataFrame({
            "val_id": val_ids,
            "target": targets,
            **predictions_by_fold,
        })

        predictions_df.to_csv(predictions_filename, index=False)

        results = []

        for fold in self.checkpoints:

            predictions = predictions_df[f"fold_{fold:02d}_pred"]
            metrics = self.calculate_classification_metrics(targets, predictions)

            results.append({
                "target_sub_exp": target_name,
                "fold": fold,
                **metrics,
            })

        return results

    def cross_evaluate(
        self,
        source_name: str,
        target_dms: dict[str, ml_core.BaseDataModule],
    ) -> list[dict]:


        cross_eval_dir = os.path.join(self.base_dir, "cross_eval")
        os.makedirs(cross_eval_dir, exist_ok=True)

        metrics_filename = os.path.join(cross_eval_dir, "metrics.csv")

        if os.path.isfile(metrics_filename):

            print(
                f"Skipping cross evaluation: "
                f"{metrics_filename} already exists."
            )

            return pd.read_csv(metrics_filename).to_dict("records")

        results = []

        for target_name, target_dm in target_dms.items():

            if target_name == source_name:
                continue

            target_results = self.cross_evaluate_target(
                target_dm=target_dm,
                target_name=target_name,
            )

            results.extend(target_results)

        metrics_df = pd.DataFrame([
            {
                "source_sub_exp": source_name,
                **result,
            }
            for result in results
        ])

        metrics_df.to_csv(metrics_filename, index=False)
        return results

    @staticmethod
    def compile_experiment_results(exp_dir, sub_experiments):
        """Compile mean/std metrics into evaluation-vs-training tables."""

        metrics = ["acc", "recall", "f1"]

        for metric in metrics:

            filename = os.path.join(exp_dir, f"{metric}.csv")
            heatmap_filename = os.path.join(exp_dir, f"{metric}.png")

            if os.path.isfile(filename) and os.path.isfile(heatmap_filename):
                print(
                    f"Skipping compilation of '{metric}': "
                    f"results already exist."
                )
                continue

            values = pd.DataFrame(index=sub_experiments, columns=sub_experiments, dtype=float)
            formatted = pd.DataFrame(index=sub_experiments, columns=sub_experiments, dtype=str)

            for source_name in sub_experiments:

                source_dir = os.path.join(exp_dir, source_name)

                internal_filename = os.path.join(source_dir, "metrics.csv")
                internal_df = pd.read_csv(internal_filename)

                cross_filename = os.path.join(source_dir, "cross_eval", "metrics.csv")
                cross_df = pd.read_csv(cross_filename)

                for target_name in sub_experiments:

                    if target_name == source_name:
                        df = internal_df
                    else:
                        df = cross_df[cross_df["target_sub_exp"] == target_name]

                    mean = df[metric].mean()
                    std = df[metric].std()

                    values.loc[target_name, source_name] = mean

                    formatted.loc[target_name, source_name] = (
                        f"{mean * 100:.1f} +- {std * 100:.1f} %"
                    )

            filename = os.path.join(exp_dir, f"{metric}.csv")
            formatted.to_csv(filename)

            heatmap_filename = os.path.join(exp_dir, f"{metric}.png")
            _save_heatmap(values, formatted, filename=heatmap_filename)
