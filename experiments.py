from copy import deepcopy
import math
from config import DEFAULT_CONFIG
from simulation import run_simulation

# t-critical for 95% confidence, df = 19 (n=20)
T_95 = 2.093

# Helper: Confidence Interval

def ci_95(samples):
    n = len(samples)
    mean_val = sum(samples) / n
    variance = sum((x - mean_val) ** 2 for x in samples) / (n - 1)
    std = math.sqrt(variance)
    margin = T_95 * std / math.sqrt(n)
    return mean_val, mean_val - margin, mean_val + margin

# Run ONE configuration for multiple replications

def run_configuration(config_name, prep_rooms, recovery_rooms, seeds):
    prep_queue_results = []
    blocking_results = []
    recovery_full_results = []

    for seed in seeds:
        cfg = deepcopy(DEFAULT_CONFIG)
        cfg.num_prep_rooms = prep_rooms
        cfg.num_recovery_rooms = recovery_rooms
        cfg.random_seed = seed
        cfg.simulation_time = 1000

        monitor = run_simulation(cfg)

        total_time = monitor.theatre_utilization.total_time

        avg_queue = monitor.prep_queue_stats.get_time_weighted_average(total_time)
        block_prob = monitor.get_blocking_statistics()['blocking_probability']
        rec_full_prob = monitor.get_recovery_full_probability()

        prep_queue_results.append(avg_queue)
        blocking_results.append(block_prob)
        recovery_full_results.append(rec_full_prob)

    print(f"\n====== RESULTS FOR {config_name} ======")
    print("Prep Queue Length 95% CI:", ci_95(prep_queue_results))
    print("OR Blocking Probability 95% CI:", ci_95(blocking_results))
    print("All Recovery Full Probability 95% CI:", ci_95(recovery_full_results))

    return {
        'queue': prep_queue_results,
        'blocking': blocking_results,
        'recovery_full': recovery_full_results
    }


# MAIN EXECUTION

if __name__ == "__main__":

    print("\n=== ASSIGNMENT 3 EXPERIMENTS ===")

    # INDEPENDENT EXPERIMENTS
    print("\n--- Independent Runs ---")

    seeds_3p4r = [100 + i for i in range(20)]
    seeds_3p5r = [400 + i for i in range(20)]
    seeds_4p5r = [800 + i for i in range(20)]

    ind_3p4r = run_configuration("3p4r", 3, 4, seeds_3p4r)
    ind_3p5r = run_configuration("3p5r", 3, 5, seeds_3p5r)
    ind_4p5r = run_configuration("4p5r", 4, 5, seeds_4p5r)


    # PAIRED EXPERIMENTS
    print("\n--- Paired Runs ---")

    common_seeds = [2000 + i for i in range(20)]

    paired_3p4r = run_configuration("3p4r (paired)", 3, 4, common_seeds)
    paired_3p5r = run_configuration("3p5r (paired)", 3, 5, common_seeds)

    # Differences per replication
    block_differences = [
        a - b for a, b in zip(paired_3p5r['blocking'], paired_3p4r['blocking'])
    ]

    print("\nPaired Difference (3p5r - 3p4r) - OR Blocking CI:")
    print(ci_95(block_differences))


    print("\n=== END OF EXPERIMENTS ===")
