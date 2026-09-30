#!/usr/bin/env python3
"""Reproduces the scorer-validation numbers of Sec. 3.3.

Two LLM judges graded every Probe B definition in the study (results/census_verdicts.csv),
blind to artifact, level and Tier 2 verdict, under the rubric in code/judge_rubric.md:
gpt-oss-20b (gpt-oss:20b), the primary judge, and Gemma 4 (gemma4:31b-cloud), which
replicates it. This script compares Tier 2 with each judge, and each judge with the
human-verified sample (results/definition_human_verification.csv).

    python code/census_check.py
"""
import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results")
d = pd.read_csv(os.path.join(R, "census_verdicts.csv"))
PRIMARY, SECOND = "gptoss20b", "gemma4"
J = {PRIMARY: "gpt-oss-20b", SECOND: "Gemma 4"}


def kappa(a, b):
    po = (a == b).mean()
    pe = a.mean()*b.mean() + (1-a.mean())*(1-b.mean())
    return (po-pe)/(1-pe)


cell = d.groupby(["config", "model"])[["t2", *J]].mean().reset_index()
print(f"Census: {len(d):,} definitions from {len(cell)} file-level evaluations\n")

# Relative drop of every quantized evaluation from its own full-precision baseline.
base = cell.model.str.contains("fp16|bf16|f16", case=False)
rows = []
for cfg, g in cell.groupby("config"):
    b = g[base.loc[g.index]]
    if len(b) != 1 or (b[["t2", *J]].iloc[0] == 0).any():
        continue                      # needs a single, non-zero full-precision baseline
    b = b.iloc[0]
    for _, q in g[~base.loc[g.index]].iterrows():
        rows.append({c: 100*(q[c]-b[c])/b[c] for c in ["t2", *J]})
drops = pd.DataFrame(rows)

# The eight sub-2B artifacts: baseline (all eight) and Q2 (the seven with a Q2 build).
c = d[d.cohort_artifact.notna()]


def judge_block(j):
    lab = J[j]
    gap = 100*(cell[j] - cell.t2)
    print(f"   Tier 2 scores below {lab} in {(gap > 0).sum()} of {len(cell)} evaluations,"
          f" {gap.mean():.1f} points below on average")
    e = drops.t2 - drops[j]
    print(f"   {len(drops)} quantized evaluations: Tier 2 drop lies {e.abs().mean():.1f} points from"
          f" {lab}'s on average (median {e.abs().median():.1f}; {100*(e.abs() <= 5).mean():.1f}%"
          f" within 5 points; r = {drops.t2.corr(drops[j]):.3f}; mean signed error {e.mean():+.1f})")
    for lvl, name in [("base", "baseline"), ("q2", "Q2")]:
        x = c[c.cohort_level == lvl]
        fn = int(((x.t2 == 0) & (x[j] == 1)).sum()); fp = int(((x.t2 == 1) & (x[j] == 0)).sum())
        print(f"   sub-2B {name:8s}: {100*x[x.t2 == 0][j].mean():.1f}% of Tier 2 rejections are false"
              f" negatives; precision {100*x[x.t2 == 1][j].mean():.1f}%  ({fn} false negatives,"
              f" {fp} false positives)")


print("1. Primary judge: gpt-oss-20b")
judge_block(PRIMARY)
g = c.groupby(["cohort_artifact", "cohort_level"])[["t2", PRIMARY]].mean().unstack()
shifts = []
for art in g.index:
    top = "q3" if pd.notna(g.loc[art, ("t2", "q3")]) else "q2"   # Granite-4.0-H-350M: Q3
    dr = {j: 100*(g.loc[art, (j, top)] - g.loc[art, (j, "base")])/g.loc[art, (j, "base")]
          for j in ["t2", PRIMARY]}
    shifts.append((art, top, dr["t2"], dr[PRIMARY]))
print("   sub-2B drops at the most aggressive level, Tier 2 vs gpt-oss-20b:")
for art, top, a, b in sorted(shifts, key=lambda r: r[2]):
    print(f"     {art:20s} {top}  Tier 2 {a:6.1f}%  gpt-oss {b:6.1f}%")
print(f"   larger under gpt-oss in {sum(b < a for _, _, a, b in shifts)} of {len(shifts)};"
      f" none moves by more than {max(abs(b - a) for _, _, a, b in shifts):.1f} points")

print("\n2. Second judge: Gemma 4")
print(f"   agrees with gpt-oss-20b on {100*(d[SECOND] == d[PRIMARY]).mean():.1f}% of definitions,"
      f" Cohen's kappa {kappa(d[SECOND], d[PRIMARY]):.2f}")
judge_block(SECOND)

print("\n3. Human verification (results/definition_human_verification.csv)")
h = pd.read_csv(os.path.join(R, "definition_human_verification.csv"))
h = h.merge(d[["config", "model", "word", "output", *J]].rename(columns={"output": "census_output"}),
            on=["config", "model", "word"], how="left")
assert h[PRIMARY].notna().all() and (h.output == h.census_output).all()
print(f"   {len(h)} definitions, all in the census with identical output text;"
      f" strata {h.groupby(['stratum', 'level']).size().to_dict()}")
print(f"   Tier 2 rejections: {(h.t2 == 0).sum()} of {len(h)}"
      f" ({100*(h.t2 == 0).mean():.1f}%, against {100*(d.t2 == 0).mean():.1f}% in the full set)")
for j, lab in J.items():
    print(f"   {lab:12s} agrees with the human on {(h[j] == h.human_verdict).sum()} of {len(h)},"
          f" Cohen's kappa {kappa(h[j], h.human_verdict):.2f}")

print("\n4. Fig. 1: Granite-4.0-H-1B, 'dirge'")
f = d[(d.word == "dirge") & d.config.str.startswith("Experiment-34-")]
print(f[["model", "t2", *J]].to_string(index=False))
