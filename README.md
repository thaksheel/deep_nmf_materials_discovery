# **Deep NMF for Materials Discovery**

A unified research codebase for benchmarking  **Nonnegative Matrix Factorization (NMF)** ,  **deep NMF** ,  **multilayer NMF** ,  **hierarchical rank‑2 NMF** ,  **semi‑supervised NMF** , and **min‑volume deep NMF** on high‑dimensional materials datasets.

This repository accompanies the experiments in the paper and provides a fully reproducible pipeline.

## **📦 Repository Structure**

text

```
.
├── data/                         # Heuslerene dataset (composition + descriptors)
├── nmfs/                         # Standalone NMF interface (algorithms only)
├── src/
│   ├── core/                     # Core utilities and shared infrastructure
│   │   ├── finetune/             # Fine‑tuning utilities for NMF models
│   │   ├── sensitivity/          # Robustness & noise‑perturbation analysis
│   │   ├── utils/                # Logging, normalization, helpers
│   │   ├── init/                 # NMF initialization schemes
│   │   └── algorithms/           # NMF algorithm selection & wrappers
│   │
│   ├── deep/                     # PyTorch implementations
│   │   ├── multilayer/           # Multilayer NMF (layer‑wise factorization)
│   │   ├── deep_nmf/             # Jointly optimized deep NMF
│   │   └── minvol/               # Min‑volume deep NMF (identifiable)
│   │
│   ├── standard/                 # Classical NMF methods (NumPy)
│   │   ├── beta_nmf.py
│   │   ├── frobenius_nmf.py
│   │   └── hierarchical_rank2.py
│   │
│   ├── supervised/               # Random Forest + evaluation metrics
│   ├── interpretation/           # SHAP, element attribution, NMI, Jaccard, entropy
│   └── runner/                   # Full pipeline: NMF → supervised → interpretability
│
├── run_*.py                      # Scripts to reproduce all paper experiments
├── data_analysis.py              # Dataset exploration utilities
├── evaluations.py                # Evaluation helpers
├── periodic_map.py               # Element → periodic table mapping
└── README.md
```

## **🚀 Key Features**

* Multiple NMF variants:
  * β‑NMF
  * Frobenius NMF
  * Hierarchical rank‑2 NMF
  * Multilayer NMF
  * Deep NMF
  * Min‑volume deep NMF
  * Semi‑supervised NMF
* Unified pipeline:
  * NMF factorization
  * Supervised learning
  * Interpretability
  * Robustness analysis
* Interpretability tools:
  * SHAP (TreeSHAP)
  * Element‑level attribution
  * Layer‑mapping (NMI, Jaccard, entropy)
  * Min‑volume identifiability
* Reproducible experiments:
  * All paper results can be reproduced via the `run_*.py` scripts.

## **📘 Running Experiments**

Each major experiment in the paper corresponds to a dedicated script:

| Experiment                                   | Script                               |
| -------------------------------------------- | ------------------------------------ |
| Supervised metrics (RMSE, MAE, Accuracy, F1) | `run_supervised_results.py`        |
| Depth selection (GM(k))                      | `run_depths.py`                    |
| Element‑level attribution                   | `run_element_level_attribution.py` |
| Layer mapping                                | `run_layer_mapping.py`             |
| Min‑volume identifiability                  | `run_minvol_identifiability.py`    |
| Noise‑perturbation robustness               | `run_ie.py`                        |
| Semi‑supervised NMF                         | `run_ssnmf.py`                     |
| Full pipeline                                | `run.py`/`run_d.py`              |

All scripts rely on the unified runner framework in `src/runner/`.

## **🧠 Core Concepts**

* **Standard NMF** (NumPy): interpretable additive latent factors
* **Multilayer NMF** : sequential refinement
* **Deep NMF** : jointly optimized hierarchical structure
* **Min‑volume deep NMF** : identifiable, stable chemical motifs
* **Semi‑supervised NMF** : aligns factors with supervised targets
* **Interpretability** : SHAP, element attribution, hierarchical mapping
* **Robustness** : noise‑perturbation analysis across seeds, ranks, noise levels


## **🔗 Codebase**

**GitHub:** [https://github.com/thaksheel/deep_nmf_materials_discovery](https://github.com/thaksheel/deep_nmf_materials_discovery)
