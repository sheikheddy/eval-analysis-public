#!/usr/bin/env python3
"""
Analysis of systematic differences between benchmark tasks and real tasks.
Maps messiness factors to the five systematic differences from section 7.2.1:
1. Automatic scoring
2. No interaction with other agents
3. Lax resource constraints
4. Unpunishing
5. Static environments
"""

import pandas as pd
import numpy as np
import json
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
runs_file = pathlib.Path("data/external/all_runs.jsonl")
df_runs = pd.DataFrame(
    [json.loads(line) for line in runs_file.read_text().strip().splitlines()]
)

messiness_file = pathlib.Path("data/external/messiness.csv")
df_messiness = pd.read_csv(messiness_file)

messiness_tasks_file = pathlib.Path("data/external/messiness_tasks.csv")
df_messiness_tasks = pd.read_csv(messiness_tasks_file)

print("=" * 80)
print("MAPPING MESSINESS FACTORS TO SYSTEMATIC DIFFERENCES")
print("=" * 80)

# Map messiness factors to the 5 systematic differences
systematic_differences_mapping = {
    "1. Automatic scoring": ["not purely automatic scoring", "non explicit scoring description"],
    "2. No multi-agent interaction": ["realtime coordination"],
    "3. Lax resource constraints": ["resource limited"],
    "4. Unpunishing (low consequence)": ["irreversible mistake availability", "not easily resettable"],
    "5. Static environments": ["dynamic environment"]
}

print("\nMessiness factors mapped to each systematic difference:")
for diff, factors in systematic_differences_mapping.items():
    print(f"\n{diff}:")
    for factor in factors:
        count = df_messiness[factor].sum()
        print(f"  - {factor}: {count} tasks")

# Calculate how many tasks have each systematic difference
print("\n" + "=" * 80)
print("TASK COUNTS BY SYSTEMATIC DIFFERENCE")
print("=" * 80)

for diff, factors in systematic_differences_mapping.items():
    # A task has this difference if ANY of the mapped factors is 1
    has_difference = df_messiness[factors].any(axis=1)
    count = has_difference.sum()
    percentage = (count / len(df_messiness)) * 100
    print(f"\n{diff}: {count} tasks ({percentage:.1f}%)")

# Analyze model performance on tasks with vs without each systematic difference
print("\n" + "=" * 80)
print("MODEL PERFORMANCE ANALYSIS")
print("=" * 80)

# Get agent runs only
df_agent_runs = df_runs[df_runs["alias"] != "human"].copy()

# Merge with messiness data
df_agent_runs_with_messiness = df_agent_runs.merge(
    df_messiness[["task id"] + [factor for factors in systematic_differences_mapping.values() for factor in factors]],
    left_on="task_id",
    right_on="task id",
    how="left"
)

print("\nAverage model score on tasks with vs without each systematic difference:")
print("(Lower scores = harder tasks, suggesting bigger impact on generalizability)")

results = []
for diff_name, factors in systematic_differences_mapping.items():
    # Tasks that have this difference
    tasks_with_diff = df_messiness[df_messiness[factors].any(axis=1)]["task id"].unique()
    tasks_without_diff = df_messiness[~df_messiness[factors].any(axis=1)]["task id"].unique()

    # Calculate average score
    with_diff_runs = df_agent_runs_with_messiness[df_agent_runs_with_messiness["task_id"].isin(tasks_with_diff)]
    without_diff_runs = df_agent_runs_with_messiness[df_agent_runs_with_messiness["task_id"].isin(tasks_without_diff)]

    if len(with_diff_runs) > 0 and len(without_diff_runs) > 0:
        avg_score_with = np.average(
            with_diff_runs["score_binarized"],
            weights=with_diff_runs["invsqrt_task_weight"]
        )
        avg_score_without = np.average(
            without_diff_runs["score_binarized"],
            weights=without_diff_runs["invsqrt_task_weight"]
        )

        score_gap = avg_score_without - avg_score_with
        relative_gap = (score_gap / avg_score_without) * 100 if avg_score_without > 0 else 0

        results.append({
            "Systematic Difference": diff_name,
            "Avg Score WITH": avg_score_with,
            "Avg Score WITHOUT": avg_score_without,
            "Score Gap": score_gap,
            "Relative Gap (%)": relative_gap,
            "Tasks WITH": len(tasks_with_diff),
            "Tasks WITHOUT": len(tasks_without_diff)
        })

        print(f"\n{diff_name}:")
        print(f"  Tasks WITH this difference: {len(tasks_with_diff)}")
        print(f"  Tasks WITHOUT this difference: {len(tasks_without_diff)}")
        print(f"  Avg score WITH: {avg_score_with:.3f}")
        print(f"  Avg score WITHOUT: {avg_score_without:.3f}")
        print(f"  Score gap (without - with): {score_gap:.3f}")
        print(f"  Relative performance drop: {relative_gap:.1f}%")

# Convert results to DataFrame and sort by relative gap
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values("Relative Gap (%)", ascending=False)

print("\n" + "=" * 80)
print("RANKING OF SYSTEMATIC DIFFERENCES BY IMPACT")
print("=" * 80)
print("\nRanked by relative performance drop (higher = bigger impact):")
print(df_results_sorted[["Systematic Difference", "Relative Gap (%)"]].to_string(index=False))

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Score comparison
ax1 = axes[0]
x = np.arange(len(df_results_sorted))
width = 0.35

bars1 = ax1.bar(x - width/2, df_results_sorted["Avg Score WITH"], width, label='WITH difference', alpha=0.8)
bars2 = ax1.bar(x + width/2, df_results_sorted["Avg Score WITHOUT"], width, label='WITHOUT difference', alpha=0.8)

ax1.set_xlabel('Systematic Difference')
ax1.set_ylabel('Average Model Score')
ax1.set_title('Model Performance: Tasks With vs Without Each Systematic Difference')
ax1.set_xticks(x)
ax1.set_xticklabels([d.split(". ")[1] for d in df_results_sorted["Systematic Difference"]], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Relative performance drop
ax2 = axes[1]
bars = ax2.barh(range(len(df_results_sorted)), df_results_sorted["Relative Gap (%)"])
ax2.set_yticks(range(len(df_results_sorted)))
ax2.set_yticklabels([d.split(". ")[1] for d in df_results_sorted["Systematic Difference"]])
ax2.set_xlabel('Relative Performance Drop (%)')
ax2.set_title('Impact of Each Systematic Difference on Model Performance')
ax2.grid(True, alpha=0.3)

# Color the bars by magnitude
colors = plt.cm.RdYlGn_r(df_results_sorted["Relative Gap (%)"] / df_results_sorted["Relative Gap (%)"].max())
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.tight_layout()
plt.savefig('plots/systematic_differences_analysis.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to: plots/systematic_differences_analysis.png")

# Additional analysis: Look at specific messiness factors
print("\n" + "=" * 80)
print("DETAILED FACTOR ANALYSIS")
print("=" * 80)

individual_factors = [
    "dynamic environment",
    "not purely automatic scoring",
    "resource limited",
    "irreversible mistake availability",
    "realtime coordination"
]

factor_results = []
for factor in individual_factors:
    if factor in df_messiness.columns:
        tasks_with = df_messiness[df_messiness[factor] == 1]["task id"].unique()
        tasks_without = df_messiness[df_messiness[factor] == 0]["task id"].unique()

        with_runs = df_agent_runs_with_messiness[df_agent_runs_with_messiness["task_id"].isin(tasks_with)]
        without_runs = df_agent_runs_with_messiness[df_agent_runs_with_messiness["task_id"].isin(tasks_without)]

        if len(with_runs) > 0 and len(without_runs) > 0:
            avg_with = np.average(with_runs["score_binarized"], weights=with_runs["invsqrt_task_weight"])
            avg_without = np.average(without_runs["score_binarized"], weights=without_runs["invsqrt_task_weight"])
            gap = avg_without - avg_with
            rel_gap = (gap / avg_without) * 100 if avg_without > 0 else 0

            factor_results.append({
                "Factor": factor,
                "Tasks WITH": len(tasks_with),
                "Score WITH": avg_with,
                "Score WITHOUT": avg_without,
                "Relative Gap (%)": rel_gap
            })

df_factor_results = pd.DataFrame(factor_results).sort_values("Relative Gap (%)", ascending=False)
print("\nIndividual messiness factors ranked by impact:")
print(df_factor_results.to_string(index=False))

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("""
Based on this analysis, we can see which systematic difference between
benchmark tasks and real-world tasks has the LARGEST impact on model performance.

A larger relative performance drop suggests that this difference is more important
for the generalizability of the study's results to real-world tasks.
""")

plt.show()
