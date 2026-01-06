#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/3 22:53
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   processor.py
# @Desc     :

from datasets import DatasetDict, Dataset
from pathlib import Path
from random import randint

from pipeline.preprocessor import preprocess_data

from src.configs.cfg_hf import HF_CONFIG
from src.utils.HF import HFDatasetTokeniser
from src.utils.helper import Timer
from src.utils.highlighter import lines


def process_data() -> None:
    """ Process data """
    with Timer("Processing data"):
        ds_dict: DatasetDict = preprocess_data()
        train: Dataset = ds_dict["train"].remove_columns(["Date", "Stock"])
        valid: Dataset = ds_dict["valid"].remove_columns(["Date", "Stock"])
        prove: Dataset = ds_dict["test"].remove_columns(["Date", "Stock"])
        # print(train.to_pandas())
        # print(valid.to_pandas())
        # print(prove.to_pandas())

        # Tokenization
        tokeniser = HFDatasetTokeniser(HF_CONFIG.PARAMETERS.NET_CN, Path(HF_CONFIG.FILE_PATHS.PRETRAINED_MODEL))
        print(tokeniser)
        lines()

        # Get the max length of dataset
        col: str = "Reasons"
        max_train, _, _ = tokeniser.get_length(train, col)  # 295
        max_valid, _, _ = tokeniser.get_length(valid, col)  # 303
        max_prove, _, _ = tokeniser.get_length(prove, col)  # 267
        max_len: int = max(max_train, max_valid, max_prove)
        print(max_train, max_valid, max_prove, max_len)  # 295 303 267 303
        lines()

        # Use a random sample to test tokenization
        idx: int = randint(0, len(train) - 1)
        sample: str = train[idx][col]
        print(f"{idx} Sample Text: {sample}")
        indices, tokens = tokeniser.encode(sample, padding="max_length", max_length=max_len, truncation=True)
        print(f"Tokenised Output: {indices["input_ids"]}")
        print(f"Tokens: {tokens}")
        dec_out = tokeniser.decode(indices)
        print(f"Decode output: {dec_out}")
        lines()

        # Use epoch fit to process the whole dataset
        tokenised_datasets: dict[str, Dataset] = {}
        for split_name in ["train", "valid", "test"]:
            ds: Dataset = ds_dict[split_name].remove_columns(["Date", "Stock"])
            tokenised_datasets[split_name] = tokeniser.fit(
                ds, col,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                batch_size=HF_CONFIG.PROCESSOR.BATCHES,
                tensor_type="pt",
                remove_col=True,
            )
        new_ds_dict = DatasetDict(tokenised_datasets)
        print(new_ds_dict)
        lines()
        print(new_ds_dict["train"].features)
        print(new_ds_dict["valid"].features)
        print(new_ds_dict["test"].features)
        lines()

        # Save the processed dataset locally
        new_ds_dict.save_to_disk(HF_CONFIG.FILE_PATHS.DATASETS)


if __name__ == "__main__":
    process_data()
