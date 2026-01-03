#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/2 23:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_hf.py
# @Desc     :   

from dataclasses import dataclass, field

from src.configs.cfg_base import Database, FilePaths, Punctuations


@dataclass
class HFParams:
    """ HF Config """
    DATASET_NAME: str = "SelmaNajih001/FinancialClassification"


@dataclass
class HuggingFaceConfiguration:
    """ HF Config """
    PARAMETERS: HFParams = field(default_factory=HFParams)
    DATABASE: Database = field(default_factory=Database)
    FILE_PATHS: FilePaths = field(default_factory=FilePaths)
    PUNCTUATIONS: Punctuations = field(default_factory=Punctuations)


HF_CONFIG: HuggingFaceConfiguration = HuggingFaceConfiguration()
