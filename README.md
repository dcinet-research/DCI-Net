# DCI-Net: Dynamic Causal Intervention Network
**Official PyTorch Geometric Implementation**

DCI-Net performs **strictly inductive** epidemic source identification and counterfactual policy simulation on temporal contact networks using Graph Neural Networks.

## Key Features
- **3 Datasets**: Scale-Free Synthetic, SocioPatterns (N=3,006), SFHH Hospital
- **Statistical Rigor**: Wilcoxon Signed-Rank tests + 95% Bootstrap Confidence Intervals
- **Policy Analysis**: Lockdown, Vaccination, and Social Distancing counterfactuals
- **Production Ready**: Deterministic seeding, TF32 disabled, and auto-data extraction

## Quick Start

```bash
# 1. Clone & Install
git clone [https://github.com/dcinet-research/DCI-Net.git](https://github.com/dcinet-research/DCI-Net.git)
cd DCI-Net
pip install -r requirements.txt

# 2. Dataset Placement
# - Place sg_infectious_contact_list.tgz in the root directory.
# - Place tij_SFHH.dat_.gz inside a folder named /data/.

# 3. Run full pipeline
python main.py


Note: While the raw SocioPatterns dataset contains >10,000 interaction logs, the provided pipeline extracts a high-density sub-network of 3,006 nodes (as described in the manuscript) by selecting the top 25,000 strongest contact edges to ensure model stability.
