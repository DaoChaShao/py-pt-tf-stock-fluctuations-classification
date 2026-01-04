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
from transformers import AutoTokenizer

from pipeline.preprocessor import preprocess_data

from src.configs.cfg_hf import HF_CONFIG
from src.utils.TF import HFDatasetTokeniser


def process_data() -> None:
    """ Process data """
    ds_dict: DatasetDict = preprocess_data()
    train: Dataset = ds_dict["train"].remove_columns(["Date", "Stock"])
    valid: Dataset = ds_dict["valid"].remove_columns(["Date", "Stock"])
    prove: Dataset = ds_dict["test"].remove_columns(["Date", "Stock"])
    # print(train.to_pandas())
    # print(valid.to_pandas())
    # print(prove.to_pandas())

    # Tokenization
    tokeniser = HFDatasetTokeniser(HF_CONFIG.PARAMETERS.TOKENISER, Path(HF_CONFIG.FILE_PATHS.TOKENIZER))
    print(tokeniser)
    # Get the max length of dataset
    col: str = "Reasons"
    max_train, _, _ = tokeniser.get_length(train, col)  # 295
    max_valid, _, _ = tokeniser.get_length(valid, col)  # 303
    max_prove, _, _ = tokeniser.get_length(prove, col)  # 267
    max_len: int = max(max_train, max_valid, max_prove)
    print(max_train, max_valid, max_prove, max_len)  # 295 303 267 303
    # Use a random sample to test tokenization
    idx: int = randint(0, len(train) - 1)
    sample: str = train[idx][col]
    print(f"Sample Text: {sample}")
    indices, tokens = tokeniser.fit(sample, padding="max_length", max_length=128, truncation=True)
    print(f"Tokenised Output: {indices["input_ids"]}")
    print(f"Tokens: {tokens}")
    # Use epoch fit to process the whole dataset
    outs = tokeniser.batch_fit(
        train, col,
        padding="max_length", max_length=128, truncation=True,
        tensor_type="pt", batch_size=16
    )
    print(len(outs["input_ids"]))


if __name__ == "__main__":
    process_data()
