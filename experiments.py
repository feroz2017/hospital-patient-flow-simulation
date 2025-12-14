from copy import deepcopy
import math
import numpy as np
import pandas as pd
from scipy import stats
from config import SimulationConfig
from simulation import run_simulation

T_95 = 2.093

def ci_95(samples):
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_val = sum(samples) / n
    variance = sum((x - mean_val) ** 2 for x in samples) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)
    margin = T_95 * std / math.sqrt(n) if n > 0 else 0
    return mean_val, mean_val - margin, mean_val + margin

def create_config(ia_type, ia_val, prep_dist, prep_val, rec_dist, rec_val, prep_rooms, rec_rooms, seed, sim_time=5000):
    cfg = SimulationConfig()
    cfg.num_prep_rooms = prep_rooms
    cfg.num_recovery_rooms = rec_rooms
    cfg.random_seed = seed
    cfg.simulation_time = sim_time
    
    if ia_type == "exp":
        cfg.interarrival_dist = "exponential"
        cfg.interarrival_mean = ia_val
    else:
        cfg.interarrival_dist = "uniform"
        cfg.interarrival_mean = ia_val[0]
        cfg.interarrival_param2 = ia_val[1]
    
    if prep_dist == "exp":
        cfg.prep_dist = "exponential"
        cfg.prep_time_mean = prep_val
    else:
        cfg.prep_dist = "uniform"
        cfg.prep_time_mean = 30.0
        cfg.prep_param2 = 50.0
    
    if rec_dist == "exp":
        cfg.recovery_dist = "exponential"
        cfg.recovery_time_mean = rec_val
    else:
        cfg.recovery_dist = "uniform"
        cfg.recovery_time_mean = 30.0
        cfg.recovery_param2 = 50.0
    
    return cfg

def analyze_serial_correlation():
    print("\n" + "="*70)
    print("SERIAL CORRELATION ANALYSIS")
    print("="*70)
    
    cfg = create_config("exp", 25.0, "exp", 40.0, "exp", 40.0, 4, 4, 0, 8000)
    
    num_runs = 10
    num_samples = 10
    all_series = []
    
    for run_id in range(num_runs):
        cfg.random_seed = 1000 + run_id
        monitor = run_simulation(cfg)
        
        queue_samples = monitor.queue_on_arrival_samples
        if len(queue_samples) < num_samples:
            continue
        
        indices = np.linspace(0, len(queue_samples) - 1, num_samples, dtype=int)
        series = [queue_samples[i] for i in indices]
        all_series.append(series)
    
    if len(all_series) == 0:
        print("No data collected for serial correlation analysis")
        return
    
    autocorrs = []
    max_lag = min(5, num_samples - 1)
    
    for lag in range(1, max_lag + 1):
        corrs = []
        for series in all_series:
            if len(series) > lag:
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                if not np.isnan(corr):
                    corrs.append(corr)
        if corrs:
            avg = np.mean(corrs)
            autocorrs.append((lag, avg))
    
    print(f"\nAverage Autocorrelations (across {len(all_series)} runs):")
    print("Lag\tAutocorrelation")
    print("-" * 30)
    for lag, corr in autocorrs:
        print(f"{lag}\t{corr:.4f}")
    
    return autocorrs

def run_experimental_design():
    print("\n" + "="*70)
    print("EXPERIMENTAL DESIGN (2^(6-3) = 8 experiments)")
    print("="*70)
    
    design = [
        [-1, -1, -1, -1, -1],
        [+1, -1, -1, -1, +1],
        [-1, +1, -1, -1, +1],
        [+1, +1, -1, -1, -1],
        [-1, -1, +1, +1, +1],
        [+1, -1, +1, +1, -1],
        [-1, +1, +1, +1, -1],
        [+1, +1, +1, +1, +1],
    ]
    
    factors = {
        'A': {-1: ("exp", 25.0), +1: ("exp", 22.5)},
        'B': {-1: ("exp", 40.0), +1: ("unif", None)},
        'C': {-1: ("exp", 40.0), +1: ("unif", None)},
        'D': {-1: 4, +1: 5},
        'E': {-1: 4, +1: 5}
    }
    
    num_reps = 20
    results = []
    
    for exp_id, row in enumerate(design, 1):
        A, B, C, D, E = row
        
        ia_type, ia_val = factors['A'][A]
        prep_dist = "exp" if B == -1 else "unif"
        prep_val = 40.0 if B == -1 else None
        rec_dist = "exp" if C == -1 else "unif"
        rec_val = 40.0 if C == -1 else None
        prep_rooms = factors['D'][D]
        rec_rooms = factors['E'][E]
        
        queue_lengths = []
        
        for rep in range(num_reps):
            seed = exp_id * 1000 + rep
            cfg = create_config(ia_type, ia_val, prep_dist, prep_val, rec_dist, rec_val, prep_rooms, rec_rooms, seed)
            
            monitor = run_simulation(cfg)
            total_time = monitor.theatre_utilization.total_time
            avg_queue = monitor.prep_queue_stats.get_time_weighted_average(total_time)
            queue_lengths.append(avg_queue)
        
        mean_q, ci_low, ci_high = ci_95(queue_lengths)
        
        result = {
            'exp': exp_id,
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
            'mean_queue': mean_q,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'queue_lengths': queue_lengths
        }
        results.append(result)
        
        print(f"\nExperiment {exp_id}: A={A}, B={B}, C={C}, D={D}, E={E}")
        print(f"  Mean queue length: {mean_q:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    
    return results

def build_regression_model(results):
    print("\n" + "="*70)
    print("REGRESSION ANALYSIS")
    print("="*70)
    
    df = pd.DataFrame(results)
    X = df[['A', 'B', 'C', 'D', 'E']].values
    y = df['mean_queue'].values
    
    X_interactions = []
    names = ['A', 'B', 'C', 'D', 'E']
    
    for row in X:
        feats = list(row)
        feats.extend([
            row[0] * row[1], row[0] * row[2], row[0] * row[3], row[0] * row[4],
            row[1] * row[2], row[1] * row[3], row[1] * row[4],
            row[2] * row[3], row[2] * row[4], row[3] * row[4],
        ])
        X_interactions.append(feats)
    
    names.extend(['AB', 'AC', 'AD', 'AE', 'BC', 'BD', 'BE', 'CD', 'CE', 'DE'])
    X_interactions = np.array(X_interactions)
    X_const = np.column_stack([np.ones(len(X_interactions)), X_interactions])
    
    coeffs, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
    
    y_pred = X_const @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    n = len(y)
    p = len(coeffs) - 1
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else 0
    
    se = np.sqrt(ss_res / (n - p - 1)) if (n - p - 1) > 0 else 0
    se_coeffs = se * np.sqrt(np.diag(np.linalg.pinv(X_const.T @ X_const)))
    t_stats = coeffs / se_coeffs
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    print("\nRegression Model:")
    print("=" * 70)
    print(f"R² = {r_squared:.4f}")
    print(f"Adjusted R² = {adj_r_squared:.4f}")
    print("\nCoefficients:")
    print(f"{'Term':<10} {'Coefficient':<15} {'p-value':<15} {'Significant':<15}")
    print("-" * 70)
    
    term_names = ['Intercept'] + names
    sig_terms = []
    
    for i, (name, coeff, pval) in enumerate(zip(term_names, coeffs, p_values)):
        sig = "Yes" if pval < 0.05 else "No"
        print(f"{name:<10} {coeff:>15.6f} {pval:>15.6f} {sig:<15}")
        if pval < 0.05 and i > 0:
            sig_terms.append(name)
    
    print(f"\nSignificant factors (p < 0.05): {', '.join(sig_terms) if sig_terms else 'None'}")
    
    return {
        'coefficients': dict(zip(term_names, coeffs)),
        'p_values': dict(zip(term_names, p_values)),
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'significant_terms': sig_terms
    }

if __name__ == "__main__":
    import sys
    from io import StringIO
    
    buf = StringIO()
    old_stdout = sys.stdout
    
    try:
        sys.stdout = buf
        
        print("\n" + "="*70)
        print("EXPERIMENT DESIGN AND METAMODELING")
        print("="*70)
        
        autocorrs = analyze_serial_correlation()
        results = run_experimental_design()
        regression = build_regression_model(results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        output = buf.getvalue()
        
        with open("experiment-design-metamodelling-report.txt", "w") as f:
            f.write(output)
        
        sys.stdout = old_stdout
        print(output)
        print("\nReport saved to: experiment-design-metamodelling-report.txt")
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error: {e}")
        raise
