#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/4 22:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparer.py
# @Desc     :   

from torch.utils.data import DataLoader

from src.configs.cfg_hf import HF_CONFIG
from src.dataloaders.seq2seq_hf_loader import HFDataLoaderForClassification
from src.utils.HF import load_hf_data_as_ds_dict
from src.utils.helper import Timer
from src.utils.highlighter import lines


def prepare_dataloader() -> tuple[DataLoader, DataLoader]:
    """ Main Function """
    with Timer("Preparing Hugging Face Dataset"):
        # Load processed dataset dictionary
        ds_dict = load_hf_data_as_ds_dict(
            HF_CONFIG.FILE_PATHS.DATASETS,
            load_train=True, load_valid=True, load_test=False
        )
        print(ds_dict)
        lines()
        print("Dataset Checkpoint:")
        print(ds_dict["train"].features)
        print(ds_dict["valid"].features)
        # print(ds_dict["test"].features)
        """
        ****************************************************************
        DatasetDict({
            train: Dataset({
                features: ['PriceVariation', 'input_ids', 'token_type_ids', 'attention_mask'],
                num_rows: 10071
            })
            valid: Dataset({
                features: ['PriceVariation', 'input_ids', 'token_type_ids', 'attention_mask'],
                num_rows: 3669
            })
            test: Dataset({
                features: ['PriceVariation', 'input_ids', 'token_type_ids', 'attention_mask'],
                num_rows: 648
            })
        })
        ****************************************************************
        """

        loader_train = HFDataLoaderForClassification(
            dataset=ds_dict["train"],
            batch_size=HF_CONFIG.PROCESSOR.BATCHES,
            shuffle_state=True,
            workers=HF_CONFIG.PROCESSOR.WORKERS,
            drop_last=False
        )
        lines()
        print("DataLoader Checkpoint:")
        print(loader_train)
        # lines()
        # print(next(loader_train))
        loader_valid = HFDataLoaderForClassification(
            dataset=ds_dict["valid"],
            batch_size=HF_CONFIG.PROCESSOR.BATCHES,
            shuffle_state=False,
            workers=HF_CONFIG.PROCESSOR.WORKERS,
            drop_last=False
        )

        return loader_train, loader_valid


if __name__ == "__main__":
    prepare_dataloader()
