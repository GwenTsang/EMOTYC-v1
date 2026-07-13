# -*- coding: utf-8 -*-
"""
EMOTYC Error Analysis — Modular Pipeline
═════════════════════════════════════════
Modules:
    config          — Constants, paths, label mappings, fixed threshold
    data_loader     — Data loading, cleaning, feature engineering
    inference       — EMOTYC model inference + cached prediction loading
    metrics         — Error metrics (Hamming, Jaccard, Brier, violations)
    conditional     — Conditional error analysis (modes ↔ emotions)
    logit_analysis  — Logit distributions, calibration
    stratification  — Density / length stratification
    explainability  — RF + SHAP, univariate, bivariate, association rules
    visualization   — All plotting functions
    report          — Report generation
"""
