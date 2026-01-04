#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/1/2 23:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_hf.py
# @Desc     :   

from dataclasses import dataclass, field

from src.configs.cfg_base import Database, FilePaths, Punctuations
from src.configs.cfg_dl import DataPreprocessor, Hyperparameters


@dataclass
class HFParams:
    """ HF Config """
    DATASET: str = "SelmaNajih001/FinancialClassification"
    NET_CN: str = "google-bert/bert-base-chinese"
    NET_EN_FINANCIAL: str = "yiyanghkust/finbert-tone"


@dataclass
class HuggingFaceConfiguration:
    """ HF Config """
    DATABASE: Database = field(default_factory=Database)
    FILE_PATHS: FilePaths = field(default_factory=FilePaths)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PARAMETERS: HFParams = field(default_factory=HFParams)
    PROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    PUNCTUATIONS: Punctuations = field(default_factory=Punctuations)


HF_CONFIG: HuggingFaceConfiguration = HuggingFaceConfiguration()
