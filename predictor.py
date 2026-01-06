#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from pathlib import Path
from random import randint
from torch import load, Tensor, no_grad
from transformers import AutoConfig

from src.configs.cfg_hf import HF_CONFIG
from src.nets.seq2seq_hf import Seq2SeqHFTransformer
from src.utils import yellow
from src.utils.helper import Timer
from src.utils.HF import load_hf_data_as_ds_dict
from src.utils.highlighter import starts, lines, red, green, blue
from src.utils.PT import item2tensor


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
            lines()

            # Randomly select a data point for prediction
            idx: int = randint(0, len(ds_dict["test"]) - 1)
            sample: dict[str, list] = ds_dict["test"][idx]
            # print(sample)
            # lines()
            sample: dict[str, Tensor] = {
                key: item2tensor(value, embedding=True, accelerator=HF_CONFIG.HYPERPARAMETERS.ACCELERATOR).unsqueeze(0)
                for key, value in sample.items()
            }
            # print(sample)
            # lines()
            label: Tensor = sample.pop("PriceVariation")
            # print(sample)
            # print(label)
            # lines()

            # Prediction
            with no_grad():
                logits: Tensor = model(**sample)
                prediction: Tensor = logits.argmax(-1).unsqueeze(1).cpu()
                # print(prediction)
                # lines()

                starts()
                print(f"Evaluation Results for Pretrained Model - {HF_CONFIG.PARAMETERS.NET_CN}:")
                lines()
                # ----------------------------------------------------------------
                print(f"{idx} Sample Original Label:    {yellow(label.squeeze().item())}")
                print(f"{idx} Sample Predicted Label:   {blue(prediction.squeeze().item())}")
                print(f"{idx} Sample Prediction Result: {green("Bingo") if label == prediction else red("Wrong")}")
                starts()
                """
                ****************************************************************
                Evaluation Results for Pretrained Model - google-bert/bert-base-chinese:
                ----------------------------------------------------------------
                109 Sample Original Label:               1
                109 Sample Predicted Label:              0
                109 Sample Prediction Result: Wrong
                ****************************************************************
                ****************************************************************
                Evaluation Results for Pretrained Model - google-bert/bert-base-chinese:
                ----------------------------------------------------------------
                52 Sample Original Label:               0
                52 Sample Predicted Label:              1
                52 Sample Prediction Result: Wrong
                ****************************************************************
                ****************************************************************
                Evaluation Results for Pretrained Model - google-bert/bert-base-chinese:
                ----------------------------------------------------------------
                428 Sample Original Label:               1
                428 Sample Predicted Label:              1
                428 Sample Prediction Result: Bingo
                ****************************************************************
                ****************************************************************
                Evaluation Results for Pretrained Model - google-bert/bert-base-chinese:
                ----------------------------------------------------------------
                40 Sample Original Label:               0
                40 Sample Predicted Label:              0
                40 Sample Prediction Result: Bingo
                ****************************************************************
                """
        else:
            print(f"Model {params.name} does {red("NOT")} exist!")


if __name__ == "__main__":
    main()
