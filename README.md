# Replication package

Data and code for *"Saying, Not Knowing: Aggressively GGUF-Quantized Small Language Models
Still Write Rare Words They Can No Longer Define."*

27 instruction-tuned quantized artifacts (13 families, four architecture backbones,
0.35B–14B) evaluated across their published GGUF ladders on 429 frequency-validated rare
English words, under two probes: Probe A (surface inclusion of a prompt-supplied word) and
Probe B (one-sentence definition).

| | |
|---|---|
| File-level Probe-B evaluations | 257 (221 rare-word, 36 common-word) |
| Primary-artifact ladder files | 216 |
| Word-level definition outcomes | 98,409 |
| Probe-A inclusion generations | 295,227 |
| Perplexity ladders | 9, over 8 models |

## Data

| File | Contents |
|---|---|
| `data/rare_words_429.csv` | Rare-word stimulus set with target synonyms. |
| `data/common_words_100.csv` | Common-word control set, with Zipf frequencies. |
| `data/wordlist_zipf_validation.csv` | Per-word `zipf`, WordNet membership and gloss, `web2` membership, `verified_real`. All 429 words are attested by at least one source independent of the model that generated the list; the six zero-Zipf words are in both WordNet and `web2`. |
| `data/artifact_tags.csv` | Registry tag pulled for every one of the 39 evaluated configurations, with build provider (Ollama-native or Unsloth via `hf.co` passthrough). 13 native / 14 Unsloth among the 27 primary artifacts. |

## Outcomes

| File | Contents |
|---|---|
| `results/probeB_rescored_long.csv` | All 98,409 word-level definition outcomes at all three scorer tiers. `config, model, word, t0, t1, t2`. |
| `results/probeA_long.csv` | All 295,227 Probe-A outcomes, three runs per word. `included` is case-insensitive word-boundary presence of the target; `n_occurrences` is its word-boundary count in the generation, and is the basis for Table 1's repetition footnote on Qwen2.5-0.5B (≥3 occurrences in 43% of Q2 outputs against 25% at FP16). |
| `results/master_results_v2.csv` | Per-file accuracies for both probes; `pA` is the Probe-A word-level Wilcoxon p used in the Benjamini–Hochberg family. Probe-B p-values are in `bh_family.csv` and `onset_per_file.csv`. |
| `results/table2_q2_deltas.csv` | Table 1 in machine-readable form; `pB` is Table 1's McNemar p and `pA` the word-level Wilcoxon p at the same endpoint. Includes `Llama3.2-1B-alt`, the Unsloth build of the build-source control; filter it out for per-artifact claims. |
| `results/onset_per_file.csv` | One row per non-baseline rare-word file (193), with level, sub-variant, discordant counts, McNemar p and its Benjamini–Hochberg value. |
| `results/onset_heatmap_data.csv` | Per-level definitional change for all 27 artifacts, worst published sub-variant per level. |
| `results/mcnemar_all.csv` | Discordant counts and McNemar p for each Q2 comparison. |
| `results/interaction_all_rows.csv`, `results/interaction_tests_clustered.csv` | Probe×quantization interaction coefficients (word-clustered logistic), and the pre-specified Holm family. Coefficients are rounded here; the paper prints some p-values more precisely than this file stores them, and they are reproducible from `probeA_long.csv` and `probeB_rescored_long.csv` — regress the pooled binary outcome on `probe * quant` with standard errors clustered on `word`. The three Regime-I rows are null because the fit is not identified when a build scores zero on both probes; Gemma-3n-E4B's coefficient is likewise an artifact of its Probe-A baseline of 1.000, and no claim in the paper rests on it. |
| `results/bh_family.csv` | The study-wide Benjamini–Hochberg family: every non-baseline file under both probes, rare and common words (Probe A word-level Wilcoxon; Probe B McNemar under the conventions below), 436 paired comparisons, 127 significant raw, 95 surviving at q=0.05. |
| `results/zipf_gradient_regime2.csv` | Loss probability by Zipf tercile and the logistic Zipf coefficient, for the six artifacts with a significant gradient. |
| `results/common_control_expansion_stats.csv`, `results/59_…`–`64_common_defs_t2.csv` | Common-word control on the sub-2B artifacts. Report `int_coef_rare_coded` (negative = rare loses more). |
| `results/46_common_defs_t2.csv`, `results/47_common_defs_t2.csv` | Common-word controls for Granite-4.0-H-1B and LFM2.5-1.2B. |
| `results/family_scale_ladders.csv` | Within-family scale comparison for the six families publishing more than one size on a fixed tokenizer (18 artifacts, 129 files), Sec. 4.7. |
| `results/perplexity_ladder.csv` | WikiText-2 perplexity ladders, `llama-perplexity`, 100 chunks × 512 tokens. |

## Scorer validation

| File | Contents |
|---|---|
| `results/census_verdicts.csv` | **The LLM-judge census**: all 98,409 Probe-B definitions, one row each. `config, model, word` join onto `probeB_rescored_long.csv`; `output` is the definition the model wrote; `t2` is its Tier-2 score; `gptoss20b` and `gemma4` are the verdicts (1 correct, 0 incorrect) of the primary judge, gpt-oss-20b (`gpt-oss:20b`), and the replication judge, Gemma 4 (`gemma4:31b-cloud`). Each judge saw only the word and the definition, in shuffled order, and graded under `code/judge_rubric.md`. `cohort_artifact` and `cohort_level` mark the 6,864 items behind the error rates of Sec. 3.3: the eight sub-2B artifacts × 429 words at baseline and at their most aggressive level (`q2`; `q3` for Granite-4.0-H-350M, which publishes no Q2 build). |
| `results/definition_human_verification.csv` | Human verification of 65 definitions against three dictionaries (Merriam-Webster, Dictionary.com, Cambridge), blind to the scorer's verdict. A stratified sample over regime stratum (`II/IIb` or `III`, Table 1), level (`base`, `q2`) and Tier-2 verdict, so rejections are over-represented: 30 of 65, against 21.5% in the full set. `config, model, word` join onto the census; `human_verdict` is 1 or 0 (54 correct, 11 incorrect). Each judge agrees with the human on 62 of 65 (Cohen's κ = 0.83). |
| `results/synonym_key_human_verification.csv` | Human spot-check of 20 randomly sampled entries of the multi-synonym key against dictionary definitions, under the OR criterion `rescore.py` matches on. `human_verdict` is 1 or 0 per entry; all 20 are 1. |

## Raw outputs

`raw/experiment-34/` holds the definition and generation outputs for Granite-4.0-H-1B
across its full ladder — the artifact quoted verbatim in Fig. 1 and the worked example of
Sec. 4.3. Every definition from every other configuration is in `results/census_verdicts.csv`.

## Code

| File | Purpose |
|---|---|
| `code/rescore.py` | The tiered Probe-B scorer. Tier 2 is the paper's primary metric. Run with no argument to score `raw/experiment-34/`; it reproduces those 4,719 released outcomes exactly at all three tiers. |
| `code/multi_synonyms.py` | The multi-synonym answer key used by Tier 2 (mean ≈7 acceptable terms per word, counting the original synonym). |
| `code/verify_wordlist.py` | Regenerates the WordNet/`web2` columns of `wordlist_zipf_validation.csv`. |
| `code/census_check.py` | Reproduces every scorer-validation number of Sec. 3.3: Tier 2 against the primary judge and the replication judge (level gap, drop agreement, false negatives and precision in the sub-2B artifacts), the human-verification sample, and the Fig. 1 verdicts. |
| `code/judge_rubric.md` | The census rubric, sent verbatim as the system prompt to both judges. |
| `code/run_judge.py` | The resumable judge runner that produced the census (Ollama, temperature 0). |

```bash
pip install pandas numpy scipy statsmodels nltk
python code/census_check.py
python code/rescore.py > /dev/null
```

To re-judge the census with another model, build the blind file (item order shuffled, as in
the paper) and run the same rubric:

```bash
python -c "import pandas as pd; pd.read_csv('results/census_verdicts.csv')[['item_id','word','output']].sample(frac=1, random_state=0).to_csv('census_blind.csv', index=False)"
python code/run_judge.py --provider ollama --model <model> --items census_blind.csv --out verdicts_<model>.csv
```

## Conventions

1. **Relative deltas are computed from unrounded accuracies.** Table 1 displays two decimals
   but computes at full precision: `.84→.48` recomputes as −42.9%, where the true value is
   0.843823→0.475524 = −43.6%.
2. **Two McNemar conventions.** Continuity-corrected χ² by default; exact binomial where a
   discordant cell is 0, which is why the Regime-I rows print p ≈ 10⁻¹¹⁵ rather than ≈ 10⁻⁸⁵.
   The common-word comparisons use the exact form throughout.
3. **"Level" means a bit-class.** Where a level publishes sub-variants (K_S/K_M/K_L), ladder
   figures and the perplexity runs use K_M; the onset heatmap uses the worst published
   sub-variant; `onset_per_file.csv` keeps every sub-variant separately. Figs. 11–12 plot the
   worst sub-variant so that all 129 files are used; the five-of-six count Sec. 4.7 reports is
   identical under all three conventions.
4. **Tier 0 ≠ Tier 2.** The primary metric is Tier 2.

## Not included

- Probe-A generations for experiments other than 34 (every definition is in
  `census_verdicts.csv`). The inclusion outcomes are released in full, but the collapsed-output
  character-length and non-ASCII statistics of Sec. 4.2, computed on generations, cannot be
  recomputed here.
- Cryptographic digests for the probe-sweep artifacts: models were deleted after inference to
  fit a consumer storage budget. `perplexity_ladder.csv` carries `gguf_sha256` for 46 of its 51
  files; the Qwen2.5-1.5B ladder has none. Registry tags for every evaluated configuration are
  in `data/artifact_tags.csv`, and those tags are mutable upstream.

## Pipeline

Inference ran through Ollama (llama.cpp backend) on a single RTX 3060 (12 GB) and Kaggle T4
sessions, ≈155 h of measured wall-clock. LLMs from three developers appear in the
pipeline, each at a different stage: the word list was generated by Gemini 3.1 Pro (Google) and frozen
before any model was evaluated; the multi-synonym key was expanded with Claude Fable 5 (Anthropic) and
human spot-checked; and the scorer was validated over every definition by gpt-oss-20b (OpenAI)
as primary judge and Gemma 4 (Google) as replication judge, against a human-verified sample. Rarity rests on corpus statistics
alone. The paper discloses all of these as potential circularity.
Probe prompts, decoding parameters and statistical procedures are specified in Sec. 3.
