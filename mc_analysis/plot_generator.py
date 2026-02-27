from datetime import date
import time
import math
import csv
import matplotlib.pyplot as plt

from src.tree import Tree
from src.option_trade import OptionTrade
from src.market import Market
from src.trinomial_model import TrinomialModel

# Experiment parameters (tweak as needed)
S0 = 100.0
K = 102.0
r = 0.05
sigma = 0.3
div_a = 0.0
ex_div_date = date(2026, 8, 21)
pricing_date = date(2025, 9, 1)
mat_date = date(2026, 9, 1)

# Steps grid: fine at small N, coarser for larger N (convex spacing)
def convex_steps(max_steps: int = 1500, power: float = 1.6, small_cutoff: int = 30):
    steps = set(range(1, min(small_cutoff, max_steps) + 1))
    n = math.ceil(max_steps ** (1.0 / power))
    for i in range(1, n + 1):
        v = int(round(i ** power))
        if v < 1:
            v = 1
        if v > max_steps:
            v = max_steps
        steps.add(v)
    steps.add(max_steps)
    return sorted(steps)

# Choose max and generate steps
max_steps = 500
steps = convex_steps(max_steps=max_steps, power=1.7, small_cutoff=30)

# Two pruning modes to compare
pruning_configs = {
    "no_pruning": None,       
    "pruning_1e-8": 1e-8
}

results = {}

for label, threshold in pruning_configs.items():
    build_times = []
    price_times = []
    prices = []
    for N in steps:
        market = Market(underlying=S0, vol=sigma, rate=r, div_a=div_a, ex_div_date=ex_div_date)
        option_trade = OptionTrade(mat=mat_date, call_put="CALL", ex="EUROPEAN", k=K)

        # Create tree with given pruning threshold
        try:
            tree = Tree(nb_step=N, market=market, option=option_trade, pricing_date=pricing_date, prunning_threshold=threshold)
        except TypeError:
            thr_val = 0.0 if threshold is None else threshold
            tree = Tree(nb_step=N, market=market, option=option_trade, pricing_date=pricing_date, prunning_threshold=thr_val)

        # measure build time
        t0 = time.perf_counter()
        try:
            tree.build_tree()
        except Exception as exc:
            build_times.append(math.nan)
            price_times.append(math.nan)
            prices.append(math.nan)
            print(f"[{label}] N={N} build ERROR: {exc!r}")
            continue
        t_build = time.perf_counter() - t0

        # measure pricing time (model.price)
        model = TrinomialModel(pricing_date=pricing_date, tree=tree)
        t0 = time.perf_counter()
        try:
            price = model.price(option_trade)
        except Exception as exc:
            price = math.nan
            t_price = math.nan
            print(f"[{label}] N={N} price ERROR: {exc!r}")
        else:
            t_price = time.perf_counter() - t0

        build_times.append(t_build)
        price_times.append(t_price)
        prices.append(price)
        print(f"[{label}] N={N:4d} build={t_build:.3f}s price={t_price:.3f}s price_val={price:.6f}")

    results[label] = {
        "steps": steps,
        "build_times": build_times,
        "price_times": price_times,
        "prices": prices
    }

# Save CSV summary
csv_file = "timings_pruning_comparison.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["N"]
    for label in pruning_configs:
        header += [f"{label}_build_s", f"{label}_price_s", f"{label}_price"]
    writer.writerow(header)
    for idx, N in enumerate(steps):
        row = [N]
        for label in pruning_configs:
            row.append(results[label]["build_times"][idx])
            row.append(results[label]["price_times"][idx])
            row.append(results[label]["prices"][idx])
        writer.writerow(row)
print(f"Saved timings to {csv_file}")

# Plotting
plt.figure(figsize=(12, 5))

# Build times
plt.subplot(1, 2, 1)
for label in pruning_configs:
    plt.plot(steps, results[label]["build_times"], marker='o', label=label)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of steps N (log scale)")
plt.ylabel("Build time (s, log scale)")
plt.title("Tree build time")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)

# Price times
plt.subplot(1, 2, 2)
for label in pruning_configs:
    plt.plot(steps, results[label]["price_times"], marker='o', label=label)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of steps N (log scale)")
plt.ylabel("Pricing time (s, log scale)")
plt.title("Option pricing time")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
#plt.show()

# --- Same plots but in linear scale (for direct comparison) ---
plt.figure(figsize=(12, 5))

# Build times (linear)
plt.subplot(1, 2, 1)
for label in pruning_configs:
    plt.plot(steps, results[label]["build_times"], marker='o', label=label)
plt.xlabel("Number of steps N")
plt.ylabel("Build time (s)")
plt.title("Tree build time (linear scale)")
plt.legend()
plt.grid(True, ls="--", alpha=0.4)

# Price times (linear)
plt.subplot(1, 2, 2)
for label in pruning_configs:
    plt.plot(steps, results[label]["price_times"], marker='o', label=label)
plt.xlabel("Number of steps N")
plt.ylabel("Pricing time (s)")
plt.title("Option pricing time (linear scale)")
plt.legend()
plt.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
plt.show()