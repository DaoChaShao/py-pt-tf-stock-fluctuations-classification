#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, expander, caption, empty

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("Financial News (Reasons) & Stock Up and Down Predictor")
with expander("**INTRODUCTION (CN)**", expanded=False):
    caption("+ 基于HuggingFace Transformer的二分类预测系统")
    caption("+ 支持实时交互式数据抽样与模型推理")
    caption("+ 完整的训练/验证/测试工作流")
    caption("+ 自动GPU加速与模型检查点保存")
    caption("+ 早停机制与学习率调度")
    caption("+ 详细的分类指标计算与评估")
    caption("+ 模块化架构设计，易于维护扩展")
    caption("+ Streamlit交互界面，支持状态管理")

with expander("**INTRODUCTION (EN)**", expanded=True):
    caption("+ HuggingFace Transformer-based binary classification system")
    caption("+ Real-time interactive data sampling and model inference")
    caption("+ Complete train/validation/test workflow")
    caption("+ Automatic GPU acceleration and model checkpointing")
    caption("+ Early stopping and learning rate scheduling")
    caption("+ Detailed classification metrics calculation and evaluation")
    caption("+ Modular architecture design for easy maintenance and extension")
    caption("+ Streamlit interactive interface with state management")
