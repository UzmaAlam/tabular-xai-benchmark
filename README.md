# Tabular Explainable AI (XAI) & Feature Attribution Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Model%20Interpretability-orange.svg)](https://github.com/shap/shap)

## Abstract & Research Objective
This repository provides a tabular classification and post-hoc interpretability framework using TreeSHAP (SHapley Additive exPlanations). The goal is to evaluate predictive accuracy and feature attribution consistency in clinical machine learning models.

## Experimental Results
Trained on the Wisconsin Breast Cancer dataset with an 80-20 stratified split. Achieved a **ROC-AUC Score of 0.9912**.

### Global Feature Importance (SHAP Summary)
![SHAP Summary Plot](shap_summary.png)

> **Key Insight:** SHAP values visualize directional feature impact, revealing how specific clinical attributes drive model predictions toward positive or negative classes.

## ⚙️ How to Reproduce
1. Clone repository:
   ```bash
   git clone [https://github.com/UzmaAlam/tabular-xai-benchmark.git](https://github.com/UzmaAlam/tabular-xai-benchmark.git)
