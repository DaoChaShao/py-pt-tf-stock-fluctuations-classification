#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   evaluator.py
# @Desc     :

from pathlib import Path
from transformers import AutoConfig
from torch import load, device, no_grad, Tensor, argmax
from tqdm import tqdm

from src.configs.cfg_hf import HF_CONFIG
from src.dataloaders.seq2seq_hf_loader import HFDataLoaderForClassification
from src.nets.seq2seq_hf import Seq2SeqHFTransformer
from src.trainers.calc_classification import calculator_for_classification
from src.utils.HF import load_hf_data_as_ds_dict
from src.utils.helper import Timer
from src.utils.highlighter import starts, lines, green, red


def main() -> None:
    """ Main Function """
    with Timer("Next Word Prediction"):
        # Load processed dataset dictionary
        ds_dict = load_hf_data_as_ds_dict(
            HF_CONFIG.FILE_PATHS.DATASETS,
            load_train=False, load_valid=False, load_test=True
        )
        print("Dataset Checkpoint:")
        print(ds_dict)
        """
        ****************************************************************
        DatasetDict({
            test: Dataset({
                features: ['PriceVariation', 'input_ids', 'token_type_ids', 'attention_mask'],
                num_rows: 648
            })
        })
        ****************************************************************
        """
        lines()
        print(ds_dict["test"].features)
        lines()
        # Get the data
        loader_test = HFDataLoaderForClassification(
            dataset=ds_dict["test"],
            batch_size=HF_CONFIG.PROCESSOR.BATCHES,
            shuffle_state=False,
            workers=HF_CONFIG.PROCESSOR.WORKERS,
            drop_last=False
        )
        print("DataLoader Checkpoint:")
        print(loader_test)
        lines()
        # print(next(loader_test))
        # lines()

        # Load the save model parameters
        params: Path = Path(HF_CONFIG.FILE_PATHS.SAVED_NET)
        pretrained_params: Path = Path(HF_CONFIG.FILE_PATHS.PRETRAINED_MODEL)
        if params.exists() and pretrained_params.exists():
            print(f"Model {green(params.name)} and {green(pretrained_params.name)} Exists!")

            config = AutoConfig.from_pretrained(str(pretrained_params))
            model = Seq2SeqHFTransformer(config.name_or_path, config, num_labels=2)
            model.load_state_dict(load(str(params), map_location=HF_CONFIG.HYPERPARAMETERS.ACCELERATOR))
            model.to(HF_CONFIG.HYPERPARAMETERS.ACCELERATOR)
            model.eval()
            print("Model Loaded Successfully!")

            # Predict and Evaluate
            outputs: list[int] = []
            targets: list[int] = []
            with no_grad():
                for batch in tqdm(loader_test):
                    # Prepare Input data
                    batch = {key: val.to(device(HF_CONFIG.HYPERPARAMETERS.ACCELERATOR)) for key, val in batch.items()}
                    labels = batch.pop("PriceVariation")
                    logits: Tensor = model(**batch)

                    # Get probabilities and predictions
                    predictions = argmax(logits, dim=1)

                    outputs.extend(predictions.cpu().tolist())
                    targets.extend(labels.cpu().tolist())

                _metrics = calculator_for_classification(outputs, targets)
                starts()
                print(f"Evaluation Results for Pretrained Model - {HF_CONFIG.PARAMETERS.NET_CN}:")
                lines()
                for metric_name, value in _metrics.items():
                    print(f"{metric_name.capitalize()}: {value:.4f}")
                starts()
                """
                ****************************************************************
                Evaluation Results for Pretrained Model - google-bert/bert-base-chinese:
                ----------------------------------------------------------------
                Accuracy:  0.8935
                Precision: 0.8936
                Recall:    0.8935
                F1_score:  0.8935
                ****************************************************************
                """
        else:
            print(f"Model {params.name} does {red("NOT")} exist!")


if __name__ == "__main__":
    main()
