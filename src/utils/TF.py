#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/3 11:29
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   TF.py
# @Desc     :   

from datasets import (load_dataset, DatasetDict, Dataset,
                      ClassLabel,
                      load_from_disk)
from pandas import DataFrame, set_option
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerFast
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


class StratifiedColConverter:
    """ Stratified Column to ClassLabel Converter """

    def __init__(self, dataset: Dataset, column: str) -> None:
        self._dataset: Dataset = dataset
        self._col: str = column

    def _convert_new_col(self, dataset) -> Dataset:
        """ Create new stratified column with integer labels """
        out = dataset[self._col]
        dataset[self._col] = 1 if out > 0 else 0

        return dataset

    def fit(self) -> Dataset:
        """ Convert stratified column to ClassLabel"""
        # Create a new label col in place
        self._dataset = self._dataset.map(self._convert_new_col)

        # Convert to ClassLabel
        self._dataset = self._dataset.cast_column(self._col, ClassLabel(names=["Negative", "Positive"]))

        return self._dataset


def check_label_distribution(dataset_dict: DatasetDict, column: str):
    """ Check label distribution in dataset
    :param dataset_dict: dataset object
    :param column: column name
    """
    for split_name, dataset in dataset_dict.items():
        labels = dataset[column]
        total = len(labels)

        if isinstance(dataset.features[column], ClassLabel):
            class_names = dataset.features[column].names
            label_counts = {name: 0 for name in class_names}

            for label in labels:
                label_name = class_names[label]
                label_counts[label_name] += 1

            print(f"ClassLabel {split_name.upper():10s} | Total: {total:6d} | ", end="")
            for name in class_names:
                count = label_counts[name]
                print(f"{name}: {count:5d} ({count / total * 100:.1f}%) | ", end="")
            print()
        else:
            positive = sum(1 for l in labels if l == 1)
            negative = total - positive
            print(f"{split_name.upper():10s} | Total: {total:6d} | "
                  f"Positive: {positive:5d} ({positive / total * 100:.1f}%) | "
                  f"Negative: {negative:5d} ({negative / total * 100:.1f}%)")


def split_ds_dict(
        dataset: DatasetDict,
        *,
        test_size: float = 0.3, shuffle: bool = True, randomness: int = 27,
        valid_prove: bool = True,
        stratified_column: str | None = None
) -> DatasetDict:
    """ Split dataset into train and test sets
    :param dataset: dataset object
    :param test_size: test size
    :param shuffle: shuffle dataset
    :param randomness: random seed
    :param valid_prove: validation proof
    :param stratified_column: stratified column name
    :return: dataset object
    """
    ds: Dataset = dataset["train"]
    train = ds.train_test_split(
        test_size=test_size, shuffle=shuffle, seed=randomness, stratify_by_column=stratified_column
    )

    if valid_prove:
        valid = train["test"].train_test_split(
            test_size=test_size / 2, shuffle=shuffle, seed=randomness, stratify_by_column=stratified_column
        )
        return DatasetDict({"train": train["train"], "valid": valid["train"], "test": valid["test"]})
    else:
        return DatasetDict({"train": train["train"], "test": train["test"]})


class HFDatasetTokeniser:
    """ Huggingface Dataset Tokeniser """

    def __init__(self, hf_tokeniser: str, tokeniser_dir: Path | None = None) -> None:

        if tokeniser_dir:
            self._path: Path = tokeniser_dir
            self._T: PreTrainedTokenizerFast = self._init_offline_tokeniser(self._path)
        else:
            self._name: str = hf_tokeniser.lower()
            self._T: PreTrainedTokenizerFast = self._init_online_tokeniser(self._name)

    @staticmethod
    def _init_offline_tokeniser(directory: Path) -> PreTrainedTokenizerFast:
        """ Initialize offline tokeniser
        :param directory: tokeniser directory
        :return: tokeniser object
        """
        print("Tokeniser file already exists") if directory.exists() else print("Tokeniser file does NOT exist")
        return AutoTokenizer.from_pretrained(str(directory))

    @staticmethod
    def _init_online_tokeniser(hf_tokeniser: str) -> PreTrainedTokenizerFast:
        """ Initialize online tokeniser
        :param hf_tokeniser: huggingface tokeniser name
        :return: tokeniser object
        """
        return AutoTokenizer.from_pretrained(hf_tokeniser)

    def get_length(self, dataset: Dataset, column: str) -> tuple[int, int, int]:
        """ Get token length of text
        :param dataset: dataset object
        :param column: column name to tokenize
        :return: tuple of (max length, average length, min length)
        """
        outs = self._T(list(dataset[column]), truncation=False, padding=False)
        lengths: list[int] = [len(ids) for ids in outs["input_ids"]]

        return max(lengths), sum(lengths) // len(lengths), min(lengths)

    def encode(self,
               sample: str,
               *,
               padding: str | Literal["max_length", "longest"] = "max_length", max_length: int = 128,
               truncation: bool = True, tensor_type: str | Literal["pt", "tf"] | None = None
               ):
        """ Fit tokeniser on sample text
        :param sample: sample text
        :param padding: padding method, either 'max_length' or 'longest'
        :param max_length: maximum length
        :param truncation: truncation method
        :param tensor_type: tensor type, either 'pt' for PyTorch or 'tf' for TensorFlow
        :return: tokenised output
        """
        indices = self._T(
            sample,
            padding=padding, max_length=max_length,
            truncation=truncation,
            return_tensors="pt" if tensor_type == "pt" else "tf" if tensor_type == "tf" else None
        )
        tokens = self._T.convert_ids_to_tokens(indices["input_ids"])

        return indices, tokens

    def decode(self, indices, *, skip: bool = True) -> str:
        """ Decode token ids back to text """
        return self._T.decode(indices.input_ids, skip_special_tokens=skip)

    def fit(self,
            dataset: Dataset, column: str,
            *,
            padding: str | Literal["max_length", "longest"] = "max_length", max_length: int = 128,
            truncation: bool = True,
            batch_size: int | Literal[8, 16, 32, 64] | None = None,
            tensor_type: str | Literal["pt", "tf"] | None = None,
            remove_col: bool = True,
            ) -> Dataset:
        """ Fit tokeniser on entire dataset
        :param dataset: dataset object
        :param column: column name to tokenize
        :param padding: padding method, either 'max_length' or 'longest'
        :param max_length: maximum length
        :param truncation: truncation method
        :param batch_size: batch size for processing
        :param tensor_type: tensor type, either 'pt' for PyTorch or 'tf' for TensorFlow
        :param remove_col: remove column if true
        :return: tokenized dataset
        """
        return dataset.map(
            lambda batch: self._T(
                batch[column],
                padding=padding, max_length=max_length,
                truncation=truncation,
                return_tensors="pt" if tensor_type == "pt" else "tf" if tensor_type == "tf" else None
            ),
            batched=True,
            batch_size=None if batch_size is None else batch_size,
            remove_columns=[column] if remove_col else None,
        )

    def __repr__(self):
        return f"{"Offline" if self._path else "Online"} HF Tokeniser loaded successfully"


def load_hf_data_as_ds_dict(
        load_path: str | Path, *, load_train: bool = True, load_valid: bool = False, load_test: bool = False
) -> DatasetDict:
    """ Load Hugging Face dataset from local disk
    :param load_path: path to load dataset from
    :param load_train: load training set
    :param load_valid: load validation set
    :param load_test: load test set
    :return: dataset object
    """
    ds: dict[str, Dataset] = {}

    load_state: tuple[bool, bool, bool] = (load_train, load_valid, load_test)
    match load_state:
        case [True, True, True]:
            ds["train"] = load_from_disk(f"{load_path}/train")
            ds["valid"] = load_from_disk(f"{load_path}/valid")
            ds["test"] = load_from_disk(f"{load_path}/test")
        case [True, True, False]:
            ds["train"] = load_from_disk(f"{load_path}/train")
            ds["valid"] = load_from_disk(f"{load_path}/valid")
        case [False, False, True]:
            ds["test"] = load_from_disk(f"{load_path}/test")
        case [True, False, False]:
            ds["train"] = load_from_disk(f"{load_path}/train")
        case _:
            raise ValueError("At least one of load_train, load_valid, load_test must be True")

    return DatasetDict(ds)


if __name__ == "__main__":
    pass
