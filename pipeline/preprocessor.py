#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/3 21:26
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preprocessor.py
# @Desc     :   

from datasets import load_dataset, DatasetDict
from pathlib import Path

from src.configs.cfg_hf import HF_CONFIG
from src.utils.TF import (load_ds_dict_locally, summary_ds_dict,
                          split_ds_dict)


def preprocess_data() -> DatasetDict:
    """ Preprocess data """
    path: Path = Path(HF_CONFIG.FILE_PATHS.DATA_HF)

    if path.exists():
        print("HuggingFace configuration file already exists")

        # Access Hugging Face Dataset
        dataset_dict = load_ds_dict_locally(path, file_type="arrow")
        print(dataset_dict)
        # Check the dataset details
        # summary_dataset_dict(dataset_dict, "train", show_dup=False)

        # Split the data
        ds_dict = split_ds_dict(dataset_dict)
        print(ds_dict)
        # summary_ds_dict(ds_dict, "train", show_dup=False)
        # summary_ds_dict(ds_dict, "valid", show_dup=False)
        # summary_ds_dict(ds_dict, "test", show_dup=False)

        return ds_dict

    else:
        print(f"{path.name} doesn't exist!")

        ds_dict = load_dataset(HF_CONFIG.PARAMETERS.DATASET)
        print(ds_dict)

        return ds_dict


if __name__ == "__main__":
    preprocess_data()
