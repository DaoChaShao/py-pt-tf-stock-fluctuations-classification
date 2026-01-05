#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Trainers & Metrics Module - PyTorch Implementations
----------------------------------------------------------------
This module provides a complete set of specialized PyTorch trainer
classes and metric calculators for various machine learning tasks
including regression, sequence classification, semantic segmentation,
and sequence-to-sequence modeling.

Main Categories:
+ BaseTorchTrainer: An abstract base trainer defining the training loop interface.
+ AutoTorchTrainer: A universal trainer that automatically adapts to different task types.

+ TorchTrainer4Regression: Trainer for regression models such as MLP
  or CNN, providing end-to-end training loops and regression metrics

+ TorchTrainer4Seq2Classification: Trainer for sequence classification
  using RNN/GRU/LSTM architectures with full support for sequential
  data batching and classification evaluation

+ TorchTrainer4UNetSemSeg: Trainer for UNet-based semantic segmentation
  with image-mask training cycles, IoU computation, pixel accuracy,
  and confusion matrix evaluation

+ SeqToSeqTorchTrainer: Base trainer for Sequence-to-Sequence models.

+ SeqToSeqWithAttnTorchTrainer: Trainer for Seq2Seq models with attention mechanisms.

+ SeqToSeqTransformerTorchTrainer: Trainer for Transformer-based Seq2Seq models.

Utility Functions:
+ calculator_for_classification: Metrics calculation for classification
+ calculator_for_confusion_metrics: Confusion matrix-based metrics
+ calc_binary_sem_seg_iou: Binary semantic segmentation IoU and pixel accuracy
+ calculator_for_regression: Regression metrics computation
+ TextQualityScorer: Evaluator for generated text quality.

Usage:
+ Direct import of trainer classes via:
    - from src.trainers import AutoTorchTrainer, BaseTorchTrainer, UNetSemSegTorchTrainer, etc.
+ Instantiate trainer classes with model, data loaders, optimizer, and config
  to perform full supervised training workflows with built-in metrics.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.3.0"

from .auto_trainer import AutoTorchTrainer

from .base_trainer import BaseTorchTrainer

from .calc_classification import calculator_for_classification
from .calc_cm import calculator_for_confusion_metrics
from .calc_iou import calc_binary_sem_seg_iou
from .calc_regression import calculator_for_regression
from .calc_seq_text_quilty import TextQualityScorer

from .sem_seg_trainer import UNetSemSegTorchTrainer
from .seq2seq_attn_trainer import SeqToSeqWithAttnTorchTrainer
from .seq2seq_tf_trainer import SeqToSeqTransformerTorchTrainer
from .seq2seq_trainer import SeqToSeqTorchTrainer

__all__ = [
    "AutoTorchTrainer",

    "BaseTorchTrainer",

    "calculator_for_classification",
    "calculator_for_confusion_metrics",
    "calc_binary_sem_seg_iou",
    "calculator_for_regression",
    "TextQualityScorer",

    "UNetSemSegTorchTrainer",
    "SeqToSeqWithAttnTorchTrainer",
    "SeqToSeqTransformerTorchTrainer",
    "SeqToSeqTorchTrainer"
]
