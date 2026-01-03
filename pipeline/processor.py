#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/3 22:53
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   processor.py
# @Desc     :   

from datasets import DatasetDict

from pipeline.preprocessor import preprocess_data


def process_data() -> None:
    """ Process data """
    ds_dict: DatasetDict = preprocess_data()
    print(ds_dict)


if __name__ == "__main__":
    process_data()
