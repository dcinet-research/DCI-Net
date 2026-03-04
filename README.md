# DCI-Net: Dynamic Causal Intervention Network

This repository contains the official PyTorch implementation and evaluation pipeline for the proposed DCI-Net framework. 

DCI-Net is designed for strictly inductive epidemic source localization and counterfactual policy simulation on temporal contact networks.

## Requirements
To install the necessary dependencies, run:
`pip install -r requirements.txt`

## Dataset Setup
The evaluation pipeline requires two network datasets in the root directory:
1. **SFHH Hospital Data:** `tij_SFHH.dat_.gz` (continuous-time hospital ward data).
2. **SocioPatterns Data:** `sg_infectious_contact_list.tgz` (physical contact network data).

**Data Extraction Instruction (Required before running):**
Before running the pipeline, you must extract the SocioPatterns archive into a directory named `infectious_data/` in the root folder. You can extract it using your operating system's default archive tool, or via the command line:

`mkdir infectious_data`
`tar -xzf sg_infectious_contact_list.tgz -C infectious_data/`

**Using Custom Datasets:**
Researchers can easily adapt this pipeline for custom graphs. Simply replace the files inside the `infectious_data/` directory with your own temporal/static edgelists (tab-separated `.txt` files with columns `t, u, v`) and the `InfectiousLoader` class will automatically parse them.

## Running the Pipeline
To execute the full evaluation pipeline—including deterministic model training, Top-K shortlisting, Wilcoxon statistical tests, and Structural Causal Model (SCM) policy simulations—run:
`python main.py`

## Outputs
The script will sequentially output:
1. Top-K Clinical Shortlist accuracies with 95% Bootstrap Confidence Intervals.
2. Non-parametric statistical significance tests (Wilcoxon) against topological baselines.
3. SCM cascade reduction metrics comparing DCI-Net targeted lockdowns versus global social distancing policies.
4. Extended Ablation Studies.
