#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/3 21:26
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preprocessor.py
# @Desc     :   

from datasets import load_dataset, DatasetDict, ClassLabel
from pathlib import Path

from src.configs.cfg_hf import HF_CONFIG
from src.utils.HF import (load_ds_dict_locally, summary_ds_dict,
                          StratifiedColConverter,
                          split_ds_dict, check_label_distribution)
from src.utils.helper import Timer
from src.utils.highlighter import lines


def preprocess_data() -> DatasetDict:
    """ Preprocess data """
    with Timer("Preprocessing data"):
        path: Path = Path(HF_CONFIG.FILE_PATHS.DATA_HF)

        if path.exists():
            print("HuggingFace configuration file already exists")
            lines()

            # Access Hugging Face Dataset
            dataset_dict = load_ds_dict_locally(path, file_type="arrow")
            print(dataset_dict)
            # Check the dataset details
            # summary_dataset_dict(dataset_dict, "train", show_dup=False)
            lines()

            # Split the data
            col: str = "PriceVariation"
            converter = StratifiedColConverter(dataset_dict["train"], col)
            dataset_dict["train"] = converter.fit()
            # print(dataset_dict["train"].features)
            # print(dataset_dict["train"].features[col].int2str(0))
            ds_dict = split_ds_dict(dataset_dict, stratified_column=col)
            print(ds_dict)
            lines()
            # summary_ds_dict(ds_dict, "train", show_dup=False)
            # summary_ds_dict(ds_dict, "valid", show_dup=False)
            # summary_ds_dict(ds_dict, "test", show_dup=False)
            print(ds_dict["train"].features)
            print(ds_dict["valid"].features)
            print(ds_dict["test"].features)
            lines()

            check_label_distribution(ds_dict, col)
            """
            ****************************************************************
            ClassLabel TRAIN      | Total:  10071 | Negative:  5177 (51.4%) | Positive:  4894 (48.6%) |
            ClassLabel VALID      | Total:   3669 | Negative:  1886 (51.4%) | Positive:  1783 (48.6%) | 
            ClassLabel TEST       | Total:    648 | Negative:   333 (51.4%) | Positive:   315 (48.6%) | 
            ****************************************************************
            """

            return ds_dict

        else:
            print(f"{path.name} doesn't exist!")

            ds_dict = load_dataset(HF_CONFIG.PARAMETERS.DATASET)
            print(ds_dict)

            return ds_dict


if __name__ == "__main__":
    preprocess_data()
