
import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict, Counter

# Try to import deap; if unavailable, provide a friendly error.
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except Exception as e:
    DEAP_AVAILABLE = False

# ---------------------------
# Configuration
# ---------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# QIDs and Sensitive Attribute for Bot-IoT
QIDS = ['sport', 'dport', 'seq_number', 'proto', 'category']
SENSITIVE_ATTRIBUTE = 'subcategory'

# ---------------------------
# 1) Data Loading & Preprocessing
# ---------------------------
def load_botiot_csv(path):
    """
    Loads a Bot-IoT-like CSV. Requires at least the following columns:
    - sport (int), dport (int), seq_number (int), proto (str), category (str), subcategory (str)
    Additional columns are ignored.
    """
    df = pd.read_csv(path)
    missing_cols = [c for c in (QIDS + [SENSITIVE_ATTRIBUTE]) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    # Basic cleaning: drop NA in required fields
    df = df.dropna(subset=QIDS + [SENSITIVE_ATTRIBUTE]).copy()
    # Ensure dtypes
    for col in ['sport', 'dport', 'seq_number']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['sport', 'dport', 'seq_number'])
    df['sport'] = df['sport'].astype(int)
    df['dport'] = df['dport'].astype(int)
    df['seq_number'] = df['seq_number'].astype(int)
    # Normalize textual columns (lowercase trim)
    for col in ['proto', 'category', 'subcategory']:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df

# ---------------------------
# 2) Domain Generalization Hierarchies (DGHs)
# ---------------------------
def dgh_proto(level, v):
    # level 0: exact
    # level 1: class (Transport/Network/Link)
    # level 2: any protocol
    v = str(v).lower()
    if level <= 0:
        return v
    if level == 1:
        if v in ('tcp', 'udp'):
            return 'transport'
        if 'icmp' in v:
            return 'network'
        if v in ('arp',):
            return 'link'
        return 'other'
    return 'any-proto'

def dgh_category(level, v):
    # level 0: exact
    # level 1: attack/normal
    # level 2: any traffic
    v = str(v).lower()
    if level <= 0:
        return v
    if level == 1:
        return 'normal' if v == 'normal' else 'attack'
    return 'any-traffic'

def dgh_port(level, x):
    # level 0: exact
    # level 1: bucket by 1000
    # level 2: service class (well-known vs ephemeral)
    # level 3: any-port
    try:
        x = int(x)
    except:
        return 'unknown'
    if level <= 0:
        return x
    if level == 1:
        return (x // 1000) * 1000
    if level == 2:
        return 'ephemeral' if x >= 49152 else 'well-known'
    return 'any-port'

def dgh_seq(level, x):
    # level 0: exact
    # level 1: bucket by 100k
    # level 2: low/high
    # level 3: any-seq
    try:
        x = int(x)
    except:
        return 'unknown'
    if level <= 0:
        return x
    if level == 1:
        return (x // 100000) * 100000
    if level == 2:
        return 'low' if x < 1_000_000 else 'high'
    return 'any-seq'

# Maximum generalization depth per QID
MAX_LEVEL = {
    'sport': 3,
    'dport': 3,
    'seq_number': 3,
    'proto': 2,
    'category': 2,
}

# Apply generalization for a row given a per-attribute level dict
def apply_generalization_row(row, levels):
    return (
        dgh_port(levels['sport'], row['sport']),
        dgh_port(levels['dport'], row['dport']),
        dgh_seq(levels['seq_number'], row['seq_number']),
        dgh_proto(levels['proto'], row['proto']),
        dgh_category(levels['category'], row['category'])
    )

# ---------------------------
# 3) Lightweight RFD Discovery
# ---------------------------
def discover_rfds(df, lhs_attrs, rhs_attr, min_support=0.01, confidence=0.8):
    """
    Discover simple value-level RFDs: LHS -> RHS (subcategory)
    - For each combination of LHS attribute values (at low generalization), 
      compute the dominant RHS value; keep if support and confidence thresholds met.
    Returns a list of rules as dicts: {'lhs': {attr: value_or_range}, 'rhs_value': v, 'support': s, 'confidence': c}
    Note: This is a pragmatic, lightweight approach suitable for large data.
    """
    # We work at L0 generalization for LHS to find candidates quickly
    work = df[lhs_attrs + [rhs_attr]].copy()
    total = len(work)
    rules = []
    # Group by LHS values (may be huge; we limit unique combinations by binning numeric attrs lightly)
    # Apply mild binning to reduce explosion
    tmp = work.copy()
    if 'sport' in lhs_attrs:
        tmp['sport'] = tmp['sport'] // 1000 * 1000
    if 'dport' in lhs_attrs:
        tmp['dport'] = tmp['dport'] // 1000 * 1000
    if 'seq_number' in lhs_attrs:
        tmp['seq_number'] = tmp['seq_number'] // 100000 * 100000

    grp = tmp.groupby(lhs_attrs)
    for key, g in grp:
        supp = len(g) / total
        if supp < min_support:
            continue
        # dominant RHS value
        counts = g[rhs_attr].value_counts(dropna=False)
        top_val = counts.index[0]
        conf = counts.iloc[0] / len(g)
        if conf >= confidence:
            lhs_dict = {}
            if isinstance(key, tuple):
                for a, k in zip(lhs_attrs, key):
                    lhs_dict[a] = k
            else:
                lhs_dict[lhs_attrs[0]] = key
            rules.append({'lhs': lhs_dict, 'rhs_value': top_val, 'support': supp, 'confidence': conf})
    return rules

# ---------------------------
# 4) RFD Combination
# ---------------------------
def combine_rfds(rfds, max_width=3):
    """
    Combine RFDs that have same RHS and disjoint LHS attributes, up to 'max_width' LHS size.
    """
    by_rhs = defaultdict(list)
    for r in rfds:
        by_rhs[(r['rhs_value'])].append(r)

    combined = []
    # Simple pairwise and triple combinations
    for rhs_val, rules in by_rhs.items():
        n = len(rules)
        # pairs
        for i in range(n):
            for j in range(i+1, n):
                lhs_i = rules[i]['lhs']
                lhs_j = rules[j]['lhs']
                if set(lhs_i).isdisjoint(set(lhs_j)) and len(lhs_i) + len(lhs_j) <= max_width:
                    lhs = lhs_i.copy()
                    lhs.update(lhs_j)
                    combined.append({'lhs': lhs, 'rhs_value': rhs_val, 'combined_from': [rules[i], rules[j]]})
        # triples (optional, coarse)
        # we keep it light to avoid combinatorial explosion in practice

    return combined

# ---------------------------
# 5) Strategy Representation & Evaluation
# ---------------------------
def evaluate_strategy(df, levels_dict):
    """
    Given generalization levels per QID, compute:
    - Achieved k (minimum equivalence class size over QIDs)
    - Information Loss (IL) in [0,1], average normalized level per attribute
    """
    # Apply generalization
    gen_cols = ['g_sport', 'g_dport', 'g_seq', 'g_proto', 'g_category']
    gvals = df.apply(lambda r: apply_generalization_row(r, levels_dict), axis=1, result_type='expand')
    gvals.columns = gen_cols
    work = pd.concat([gvals], axis=1)

    # Compute k-anonymity across equivalence classes defined by generalized QIDs
    grp_sizes = work.groupby(gen_cols).size()
    k = int(grp_sizes.min()) if len(grp_sizes) > 0 else 0

    # Information Loss as mean(level / MAX_LEVEL)
    per_attr = []
    for a in ['sport','dport','seq_number','proto','category']:
        per_attr.append(levels_dict[a] / MAX_LEVEL[a] if MAX_LEVEL[a] > 0 else 0.0)
    IL = float(np.mean(per_attr))

    return k, IL

# ---------------------------
# 6) PSO Optimization
# ---------------------------
def pso_optimize(df, target_k=5, w_priv=0.5, w_util=0.5, particles=20, iterations=20):
    """
    Particle represents a vector of generalization levels per QID in fixed order:
    [sport, dport, seq_number, proto, category]
    Fitness to maximize: F = w_priv * P + w_util * (1 - U)
      where P = min(1, k/target_k), U = IL in [0,1]
    """
    if not DEAP_AVAILABLE:
        raise RuntimeError("DEAP is not available in this environment. Please install deap to run PSO.")

    ATTRS = ['sport','dport','seq_number','proto','category']

    # Register fitness and individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generators: random int within [0, MAX_LEVEL[a]]
    for idx, a in enumerate(ATTRS):
        toolbox.register(f"attr_{a}", random.randint, 0, MAX_LEVEL[a])

    def init_ind():
        return creator.Individual([toolbox.__getattribute__(f"attr_{a}")() for a in ATTRS])

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def fitness(ind):
        levels = {a: ind[i] for i, a in enumerate(ATTRS)}
        k, IL = evaluate_strategy(df, levels)
        P = min(1.0, k / float(target_k) if target_k > 0 else 1.0)
        F = w_priv * P + w_util * (1.0 - IL)
        return (F,)

    toolbox.register("evaluate", fitness)

    # PSO-like update (we'll implement a simple PSO using DEAP primitives)
    # Initialize population
    pop = toolbox.population(n=particles)

    # Initialize velocities
    velocities = [ [0.0]*len(ATTRS) for _ in range(len(pop)) ]
    # PSO hyperparameters
    w = 0.5   # inertia
    c1 = 1.0  # cognitive
    c2 = 1.0  # social

    # Initialize bests
    personal_best = [ind[:] for ind in pop]
    personal_best_f = [toolbox.evaluate(ind)[0] for ind in pop]
    global_best = personal_best[int(np.argmax(personal_best_f))][:]
    global_best_f = max(personal_best_f)

    for it in range(iterations):
        for i, ind in enumerate(pop):
            # Update velocity
            for d in range(len(ATTRS)):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (w * velocities[i][d]
                                    + c1 * r1 * (personal_best[i][d] - ind[d])
                                    + c2 * r2 * (global_best[d] - ind[d]))
                # Update position (discrete levels)
                ind[d] = int(round(ind[d] + velocities[i][d]))
                # Clamp to valid range
                ind[d] = max(0, min(ind[d], MAX_LEVEL[ATTRS[d]]))

            # Evaluate
            f = toolbox.evaluate(ind)[0]
            if f > personal_best_f[i]:
                personal_best[i] = ind[:]
                personal_best_f[i] = f
                if f > global_best_f:
                    global_best = ind[:]
                    global_best_f = f

    # Return best solution
    best_levels = {a: global_best[i] for i, a in enumerate(ATTRS)}
    best_k, best_IL = evaluate_strategy(df, best_levels)
    return best_levels, best_k, best_IL, global_best_f

# ---------------------------
# 7) Orchestration
# ---------------------------
def run_framework(csv_path,
                  min_support=0.01, confidence=0.85,
                  target_k=5, w_priv=0.5, w_util=0.5,
                  particles=20, iterations=20,
                  sample_n=None):
    df = load_botiot_csv(csv_path)
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)

    # Discover simple RFDS with LHS from QIDs (pairwise LHS only to keep it tractable)
    all_rfds = []
    LHS_CANDIDATES = [
        ['proto', 'dport'],
        ['proto', 'sport'],
        ['proto', 'category'],
        ['sport', 'dport'],
        ['seq_number', 'proto'],
        ['seq_number', 'dport'],
    ]
    for lhs in LHS_CANDIDATES:
        rfds = discover_rfds(df, lhs, SENSITIVE_ATTRIBUTE,
                             min_support=min_support, confidence=confidence)
        all_rfds.extend(rfds)

    combined = combine_rfds(all_rfds, max_width=3)

    # PSO search for best anonymization levels
    best_levels, best_k, best_IL, best_f = pso_optimize(
        df, target_k=target_k, w_priv=w_priv, w_util=w_util,
        particles=particles, iterations=iterations
    )

    result = {
        'records': len(df),
        'rfds_discovered': len(all_rfds),
        'rfds_combined': len(combined),
        'best_levels': best_levels,
        'best_k': int(best_k),
        'best_IL': float(best_IL),
        'best_fitness': float(best_f),
    }
    return result

# ---------------------------
# 8) CLI
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bot-IoT Anonymization Framework (RFD + PSO)")
    parser.add_argument("--csv", type=str, required=True, help="Path to Bot-IoT CSV")
    parser.add_argument("--min_support", type=float, default=0.01, help="Min support for RFD discovery")
    parser.add_argument("--confidence", type=float, default=0.85, help="Min confidence for RFD discovery")
    parser.add_argument("--target_k", type=int, default=5, help="Target k-anonymity for fitness")
    parser.add_argument("--w_priv", type=float, default=0.5, help="Weight for privacy in fitness")
    parser.add_argument("--w_util", type=float, default=0.5, help="Weight for utility in fitness")
    parser.add_argument("--particles", type=int, default=20, help="PSO particles")
    parser.add_argument("--iterations", type=int, default=20, help="PSO iterations")
    parser.add_argument("--sample_n", type=int, default=None, help="Optional row sample for quick runs")
    args = parser.parse_args()

    if not DEAP_AVAILABLE:
        print("ERROR: The 'deap' package is not available. Please install it to run PSO.")
        return

    res = run_framework(
        csv_path=args.csv,
        min_support=args.min_support,
        confidence=args.confidence,
        target_k=args.target_k,
        w_priv=args.w_priv,
        w_util=args.w_util,
        particles=args.particles,
        iterations=args.iterations,
        sample_n=args.sample_n
    )
    print("=== Bot-IoT Anonymization Framework Results ===")
    for k, v in res.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
