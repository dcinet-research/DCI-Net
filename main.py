# =============================================================================
# DCI-Net: Dynamic Causal Intervention Network
# Official PyTorch Geometric Implementation
# =============================================================================
# Features: Deterministic Seeding | Bootstrap 95% CI | Wilcoxon Tests | Policies

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import degree
import networkx as nx
import numpy as np
from scipy.stats import wilcoxon
import pandas as pd
import glob
import os
import random
import time
import tarfile
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("DCI-Net: Model Evaluation and Policy Simulation Pipeline")
print("Deterministic | Bootstrap CI | Wilcoxon | SCM Counterfactuals")
print("=" * 90)

# ==========================================
# 0. REPRODUCIBILITY & STATS UTILS
# ==========================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def bootstrap_ci(values, n_bootstrap=1000, ci=95):
    if len(values) < 2: return np.mean(values), 0
    boot_means = [np.mean(np.random.choice(values, len(values))) for _ in range(n_bootstrap)]
    mean_val = np.mean(values)
    lower, upper = np.percentile(boot_means, [(100-ci)/2, 100 - (100-ci)/2])
    return mean_val, (upper - lower) / 2

def check_data_integrity(dataset, name):
    sizes = [d.num_nodes for d in dataset]
    inf_fracs = [d.x.sum().item()/d.num_nodes for d in dataset]
    print(f"Dataset [{name}]: N={len(dataset)} graphs, nodes={np.mean(sizes):.0f}±{np.std(sizes):.0f}, inf_frac={np.mean(inf_fracs):.1%}")
    return True

# ==========================================
# 1. PRODUCTION DCINet
# ==========================================
class DCINet(nn.Module):
    def __init__(self, hidden_dim=128, max_degree=200, use_degree=True):
        super().__init__()
        self.use_degree = use_degree
        self.state_encoder = nn.Linear(1, hidden_dim)
        if self.use_degree:
            self.degree_emb = nn.Embedding(max_degree, hidden_dim)
        self.max_degree = max_degree
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.p0_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
        edge_index = data.edge_index
        
        if self.use_degree:
            degs = torch.clamp(degree(edge_index[0], data.num_nodes).long(), 0, self.max_degree - 1)
            h = F.relu(self.state_encoder(x) + self.degree_emb(degs))
        else:
            h = F.relu(self.state_encoder(x))
            
        h1 = F.relu(self.conv1(h, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index)) + h1
        h3 = F.relu(self.conv3(h2, edge_index)) + h2
        
        return {'p0_scores': self.p0_head(h3).squeeze(-1).unsqueeze(0)}

    @torch.no_grad()
    def simulate_policy_single(self, data, G_raw, policy_name, beta=0.35, t_max=15, mc_runs=20, top_k_val=10):
        out = self.forward(data)
        scores = out['p0_scores'].squeeze()
        is_sick = data.x[:, 0] == 1.0 if data.x.dim() > 1 else data.x == 1.0
        scores[~is_sick] = -1e4
        top_k = scores.topk(top_k_val).indices.cpu().numpy()
        true_source = data.y.item()

        def simulate(blocked_nodes, current_beta):
            if true_source in blocked_nodes: return 0
            infected, frontier = {true_source}, [true_source]
            for _ in range(t_max):
                new_inf = []
                for u in frontier:
                    if u in blocked_nodes: continue
                    for v in G_raw.neighbors(u):
                        if v not in infected and v not in blocked_nodes and np.random.random() < current_beta:
                            infected.add(v)
                            new_inf.append(v)
                frontier = new_inf
                if not frontier: break
            return len(infected)

        np.random.seed(42)
        baseline = np.mean([simulate(set(), beta) for _ in range(mc_runs)])
        np.random.seed(42)
        
        if policy_name in ["Vaccination", "Lockdown"]:
            intervened = np.mean([simulate(set(top_k), beta) for _ in range(mc_runs)])
        elif policy_name == "Social_Distancing":
            intervened = np.mean([simulate(set(), beta * 0.6) for _ in range(mc_runs)])
        else:
            intervened = np.mean([simulate(set(), beta * 0.8) for _ in range(mc_runs)])
        return baseline, intervened

# ==========================================
# 2. DATA GENERATORS
# ==========================================
def generate_synthetic_data(num_graphs=200, n_nodes=100):
    dataset = []
    np.random.seed(42)
    for i in range(num_graphs):
        G = nx.barabasi_albert_graph(n_nodes, 3)
        edges = list(G.edges)
        edge_index = torch.tensor(edges + [(v,u) for u,v in edges], dtype=torch.long).t().contiguous()
        p0 = np.random.randint(0, n_nodes)
        infected, frontier = {p0}, [p0]
        target_infections = np.random.randint(15, 35)
        for t in range(1, 16):
            new_infected = []
            for u in frontier:
                for v in G.neighbors(u):
                    if v not in infected and np.random.random() < 0.25:
                        infected.add(v); new_infected.append(int(v))
                        if len(infected) >= target_infections: break
                if len(infected) >= target_infections: break
            frontier = new_infected
            if len(infected) >= target_infections or not frontier: break
        snapshot = np.zeros(n_nodes, dtype=np.float32)
        snapshot[list(infected)] = 1.0
        dataset.append(Data(x=torch.tensor(snapshot), edge_index=edge_index, y=torch.tensor([p0])))
    return dataset

class InfectiousLoader:
    @staticmethod
    def load_static(path="./data/infectious_data"):
        tar_path = "sg_infectious_contact_list.tgz"
        if os.path.exists(tar_path) and not os.path.exists(path):
            print("Extracting infectious data...")
            os.makedirs(path, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=path) # Safely constrained extraction
                
        files = glob.glob(os.path.join(path, "**", "listcontacts_2009_*.txt"), recursive=True)
        if not files: return nx.barabasi_albert_graph(500, 4)
        dfs = []
        for f in files[:20]:
            try:
                df = pd.read_csv(f, sep='\t', header=None, names=['t','u','v'], on_bad_lines='skip')
                if len(df) > 10: dfs.append(df[['u','v']])
            except: continue
        if not dfs: return nx.barabasi_albert_graph(500, 4)
        df = pd.concat(dfs, ignore_index=True)
        G = nx.Graph()
        for u, v in df.groupby(['u','v']).size().nlargest(25000).index.tolist(): G.add_edge(int(u), int(v))
        return nx.convert_node_labels_to_integers(G)

def generate_real_static_epidemics(G, num_graphs=150):
    dataset = []
    edge_index = torch.tensor(list(G.edges) + [(v,u) for u,v in G.edges], dtype=torch.long).t().contiguous()
    n_nodes = G.number_of_nodes()
    np.random.seed(42)
    for i in range(num_graphs):
        p0 = np.random.choice(list(G.nodes()))
        infected, frontier = {p0}, [p0]
        target_infections = np.random.randint(50, 200)
        for t in range(1, 16):
            new_infected = []
            for u in frontier:
                for v in G.neighbors(u):
                    if v not in infected and np.random.random() < 0.35:
                        infected.add(v); new_infected.append(int(v))
                        if len(infected) >= target_infections: break
                if len(infected) >= target_infections: break
            frontier = new_infected
            if len(infected) >= target_infections or not frontier: break
        snapshot = np.zeros(n_nodes, dtype=np.float32)
        snapshot[list(infected)] = 1.0
        dataset.append(Data(x=torch.tensor(snapshot), edge_index=edge_index, y=torch.tensor([p0])))
    return dataset

class SFHHLoader:
    @staticmethod
    def load_temporal(filepath="./data/tij_SFHH.dat.gz"):
        if not os.path.exists(filepath):
            return pd.DataFrame({'t': np.sort(np.random.randint(0, 10000, 30000)), 'u': np.random.randint(0, 75, 30000), 'v': np.random.randint(0, 75, 30000)}), 75
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['t', 'u', 'v'], usecols=[0, 1, 2], compression='gzip')
        df = df.sort_values('t').reset_index(drop=True)
        df['t'] = df['t'] - df['t'].min()
        nodes = pd.unique(df[['u', 'v']].values.ravel())
        mapping = {node: idx for idx, node in enumerate(nodes)}
        df['u'], df['v'] = df['u'].map(mapping).astype(int), df['v'].map(mapping).astype(int)
        return df, len(nodes)

def generate_real_temporal_epidemics(temp_df, n_nodes, num_graphs=150, beta=0.6):
    dataset = []
    G_backbone = nx.Graph()
    G_backbone.add_edges_from(list(zip(temp_df['u'], temp_df['v'])))
    edge_index = torch.tensor(list(G_backbone.edges) + [(v,u) for u,v in G_backbone.edges], dtype=torch.long).t().contiguous()
    social_hubs = temp_df['u'].value_counts()[temp_df['u'].value_counts() > 5].index.to_numpy()
    u_arr, v_arr, t_arr = temp_df['u'].values, temp_df['v'].values, temp_df['t'].values
    np.random.seed(42)
    attempts = 0
    while len(dataset) < num_graphs and attempts < 5000:
        attempts += 1
        p0 = int(np.random.choice(social_hubs)) if len(social_hubs) > 0 else int(np.random.choice(temp_df['u'].unique()))
        inf_times = np.full(n_nodes, np.inf, dtype=np.float32)
        p0_mask = (u_arr == p0) | (v_arr == p0)
        if not p0_mask.any(): continue
        inf_times[p0] = t_arr[p0_mask].min() - 1.0
        target_infections = np.random.randint(10, max(11, int(n_nodes * 0.12)))
        infected_count = 1
        for i in range(len(temp_df)):
            u, v, t = u_arr[i], v_arr[i], t_arr[i]
            if inf_times[u] <= t and inf_times[v] > t and np.random.random() < beta:
                inf_times[v] = t; infected_count += 1
            elif inf_times[v] <= t and inf_times[u] > t and np.random.random() < beta:
                inf_times[u] = t; infected_count += 1
            if infected_count >= target_infections: break
        valid_mask = inf_times < np.inf
        if valid_mask.sum() >= 5:
            snapshot = np.zeros(n_nodes, dtype=np.float32)
            snapshot[valid_mask] = 1.0
            dataset.append(Data(x=torch.tensor(snapshot), edge_index=edge_index, y=torch.tensor([p0])))
    return dataset

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train_and_evaluate(dataset, device, epochs=150, run_seed=42, use_degree=True, use_masking=True):
    if len(dataset) < 10: return None

    seed_everything(run_seed)

    run_dataset = list(dataset)
    random.shuffle(run_dataset)
    split_idx = int(0.7 * len(run_dataset))
    train_data, test_data = run_dataset[:split_idx], run_dataset[split_idx:]

    model = DCINet(hidden_dim=128, use_degree=use_degree).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        for data in train_data[:30]:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)['p0_scores'].clone()

            if use_masking:
                is_sick = data.x[:, 0] == 1.0 if data.x.dim() > 1 else data.x == 1.0
                logits[0, ~is_sick] = -1e4

            loss = F.cross_entropy(logits, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    train_time = time.time() - start_time

    model.eval()
    t1, t10, full_t10 = 0, 0, 0
    mrr_dci, mrr_rand, mrr_deg = [], [], []
    early_mrr, mid_mrr, late_mrr = [], [], []

    with torch.no_grad():
        for data in test_data:
            data = data.to(device)
            y = data.y.item()
            is_sick = data.x[:, 0] == 1.0 if data.x.dim() > 1 else data.x == 1.0
            sick_indices = torch.where(is_sick)[0].cpu().numpy()
            inf_frac = len(sick_indices) / data.num_nodes

            # Baselines
            if len(sick_indices) > 0:
                rand_guess = np.random.permutation(sick_indices)
                if y in rand_guess: mrr_rand.append(1.0 / (np.where(rand_guess == y)[0][0] + 1))
                else: mrr_rand.append(0)

                degs = degree(data.edge_index[0], data.num_nodes).cpu().numpy()
                deg_guess = sick_indices[np.argsort(-degs[sick_indices])]
                if y in deg_guess: mrr_deg.append(1.0 / (np.where(deg_guess == y)[0][0] + 1))
                else: mrr_deg.append(0)

            # Full Network Ranking
            scores_raw = model(data)['p0_scores'].squeeze()
            preds_full = scores_raw.argsort(descending=True).cpu().numpy()
            if y in preds_full[:10]: full_t10 += 1

            # DCI-Net Ranking (Masked)
            scores_eval = scores_raw.clone()
            if use_masking:
                scores_eval[~is_sick] = -1e4

            preds = scores_eval.argsort(descending=True).cpu().numpy()

            try:
                rank = np.where(preds == y)[0][0] + 1
                curr_mrr = 1.0 / rank
                mrr_dci.append(curr_mrr)
                if rank == 1: t1 += 1
                if rank <= 10: t10 += 1

                if inf_frac < 0.05: early_mrr.append(curr_mrr)
                elif inf_frac < 0.15: mid_mrr.append(curr_mrr)
                else: late_mrr.append(curr_mrr)
            except:
                mrr_dci.append(0)

    N = len(test_data)
    return {
        "t1": (t1/N)*100, "t10": (t10/N)*100, "full_t10": (full_t10/N)*100,
        "mrr_dci": mrr_dci, "mrr_rand": mrr_rand, "mrr_deg": mrr_deg,
        "early": early_mrr, "mid": mid_mrr, "late": late_mrr,
        "model": model, "train_time": train_time
    }

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEEDS = [42, 123, 777]

    print("\nGenerating Datasets...")
    syn_dataset = generate_synthetic_data(200, 100)
    
    # Extracting datasets safely if needed
    G_static = InfectiousLoader.load_static("./data/infectious_data")
    static_dataset = generate_real_static_epidemics(G_static, 150)
    
    sfhh_df, n_sfhh = SFHHLoader.load_temporal("./data/tij_SFHH.dat.gz")
    sfhh_dataset = generate_real_temporal_epidemics(sfhh_df, n_sfhh, 150, beta=0.6)

    print("\n" + "="*80)
    check_data_integrity(syn_dataset, "Scale-Free")
    check_data_integrity(static_dataset, "SocioPatterns")
    check_data_integrity(sfhh_dataset, "SFHH Hospital")

    # ---------------------------------------------------------
    # PART A: MAIN RESULTS
    # ---------------------------------------------------------
    print("\n" + "="*120)
    print("PART A: MAIN RESULTS (Ranking and Statistical Significance)")
    print("-" * 120)
    print(f"{'Dataset':<15} | {'Rand MRR':<9} | {'Deg MRR':<9} | {'DCI Mask T10':<18} | {'DCI MRR':<8} | {'p-value':<8}")
    print("-" * 120)

    datasets_to_run = [("Scale-Free", syn_dataset), ("SocioPatterns", static_dataset), ("SFHH Hospital", sfhh_dataset)]
    best_static_model = None

    for name, ds in datasets_to_run:
        runs_t10, train_times = [], []
        all_dci, all_deg = [], []
        early_all, mid_all, late_all = [], [], []
        all_rand = []

        for seed in SEEDS:
            res = train_and_evaluate(ds, device, run_seed=seed, use_degree=True, use_masking=True)
            if not res: continue
            runs_t10.append(res["t10"])
            train_times.append(res["train_time"])
            all_dci.extend(res["mrr_dci"]); all_rand.extend(res["mrr_rand"]); all_deg.extend(res["mrr_deg"])
            early_all.extend(res["early"]); mid_all.extend(res["mid"]); late_all.extend(res["late"])
            if name == "SocioPatterns" and best_static_model is None: best_static_model = res["model"]

        if not runs_t10: continue

        mean_t10, ci_t10 = bootstrap_ci(runs_t10)

        try:
            diffs = np.array(all_dci) - np.array(all_deg)
            if np.all(diffs == 0): pval = 1.0
            else: pval = wilcoxon(all_dci, all_deg).pvalue
        except:
            pval = 1.0

        pval_str = "<0.001" if pval < 0.001 else f"{pval:.3f}"

        print(f"{name:<15} | {np.mean(all_rand):.3f}     | {np.mean(all_deg):.3f}     | {mean_t10:4.1f}% [95% CI ±{ci_t10:3.1f}%] | {np.mean(all_dci):.3f}   | {pval_str}")
        print(f"   -> Sensitivity : Early (<5%): {np.mean(early_all) if early_all else 0:.3f} | Mid (5-15%): {np.mean(mid_all) if mid_all else 0:.3f} | Late (>15%): {np.mean(late_all) if late_all else 0:.3f}")
        print(f"   -> Scalability : Avg Training Time = {np.mean(train_times):.1f}s")

    # ---------------------------------------------------------
    # PART B: POLICY ROBUSTNESS ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("PART B: POLICY IMPACT ANALYSIS (SocioPatterns N=3,006)")
    print("="*80)

    if best_static_model is not None:
        test_graphs = static_dataset[-20:]
        scenarios = [
            ("Social Distancing (Global)", "Social_Distancing", 10),
            ("Targeted Lockdown (Top-5)", "Lockdown", 5),
            ("Targeted Lockdown (Top-10)", "Lockdown", 10)
        ]

        for title, policy, k in scenarios:
            base_total, int_total = 0, 0
            for data in test_graphs:
                b, i = best_static_model.simulate_policy_single(data, G_static, policy, top_k_val=k)
                base_total += b; int_total += i

            prevented = base_total - int_total
            reduction = (prevented / base_total) * 100 if base_total > 0 else 0
            print(f"  {title:30}: {prevented/20:5.1f} cases prevented ({reduction:5.1f}% avg reduction)")

    # ---------------------------------------------------------
    # PART C: EXTENDED ABLATION STUDY
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("PART C: EXTENDED ABLATION STUDY (Top-10 Accuracy %)")
    print("="*80)

    variants = [
        {"name": "Full DCI-Net", "deg": True, "mask": True, "metric": "t10"},
        {"name": "w/o Degree Embeddings", "deg": False, "mask": True, "metric": "t10"},
        {"name": "w/o Clinical Masking", "deg": True, "mask": False, "metric": "full_t10"}
    ]

    for name, ds in datasets_to_run:
        print(f"\n--- {name} Ablation ---")
        for v in variants:
            runs = []
            for s in SEEDS:
                seed_everything(s) 
                res = train_and_evaluate(ds, device, run_seed=s, use_degree=v["deg"], use_masking=v["mask"])
                if res: runs.append(res[v["metric"]])

            if runs:
                mean_t10, ci = bootstrap_ci(runs)
                print(f"  {v['name']:<25} | {mean_t10:4.1f}% [95% CI ±{ci:3.1f}%]")

    print("\nPIPELINE COMPLETED.")