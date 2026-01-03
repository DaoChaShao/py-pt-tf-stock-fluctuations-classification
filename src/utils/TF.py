#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/3 11:29
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   TF.py
# @Desc     :   

from datasets import load_dataset, DatasetDict, Dataset
from pandas import DataFrame, set_option
from pathlib import Path
from typing import Literal

WIDTH: int = 64


def load_ds_dict_locally(path: str | Path, *, file_type: str | Literal["csv", "json", "arrow"]) -> DatasetDict:
    """ Load dataset locally
    :param path: path to dataset
    :param file_type: file type (csv, json, arrow)
    :return: dataset object
    """
    category: str = file_type.lower()

    match category:
        case "csv":
            return load_dataset(category, data_files=str(path))
        case "json":
            return load_dataset(category, data_files=str(path))
        case "arrow":
            return load_dataset(category, data_files=str(path))
        case _:
            raise TypeError(f"Unsupported file type of Hugging Face: {category}")


def summary_ds_dict(dataset: DatasetDict, split: str | Literal["train", "valid", "test"], show_dup: bool = False):
    """ Summary dataset
    :param dataset: dataset object
    :param split: split to show
    :param show_dup: show duplicate rows
    :return: None
    """
    df: DataFrame = dataset[split.lower()].to_pandas()

    print("*" * WIDTH)
    print(f"{split.capitalize()} Data Summary:")
    print("-" * WIDTH)
    print(df.describe())
    print("-" * WIDTH)
    print(df.head())
    print("-" * WIDTH)
    print(f"Non-Nan Amount:\n{df.notna().sum()}")
    print("-" * WIDTH)
    print(f"Nan Amount:\n{df.isna().sum()}")
    print("-" * WIDTH)
    print(f"Non-Duplicated Amount:\n{df.nunique()}")
    print("-" * WIDTH)
    print(f"Duplicated Amount: {df.duplicated().sum()}")
    if show_dup:
        set_option("display.max_rows", None)
        set_option("display.max_columns", None)
        set_option("display.max_colwidth", None)
        set_option("display.width", 2000)
    print(df[df.duplicated(keep=False)])
    print("*" * WIDTH)
    print()

    return df


def split_ds_dict(
        dataset: DatasetDict, *, test_size: float = 0.4, randomness: int = 27, valid_prove: bool = True
) -> DatasetDict:
    """ Split dataset into train and test sets
    :param dataset: dataset object
    :param test_size: test size
    :param randomness: random seed
    :param valid_prove: validation proof
    :return: dataset object
    """
    ds: Dataset = dataset["train"]
    train = ds.train_test_split(test_size=test_size, shuffle=True, seed=randomness)

    if valid_prove:
        valid = train["test"].train_test_split(test_size=test_size / 2, shuffle=True, seed=randomness)
        return DatasetDict({"train": train["train"], "valid": valid["train"], "test": valid["test"]})
    else:
        return DatasetDict({"train": train["train"], "test": train["test"]})


if __name__ == "__main__":
    pass
