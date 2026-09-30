# PIAtlas and PICorpus: code and data

This repository accompanies the submission "SoK: PIAtlas: Decomposing the Unbounded Prompt
Injection Attack Space". It contains the PIAtlas specification, the labeling prompts and runs, the
8,663 construction-labeled templates, the calibration and verification data, the crosswalk of
prior taxonomies, and the scripts that regenerate the paper's tables and figure.

## Setup

1. Run `sh unpack_data.sh` once from the repository root. The review mirror cannot serve files
   larger than 8 MB, so two JSON inputs are stored as `.json.gz`, and the script unpacks them in
   place.
2. The analysis scripts need Python 3.12 or newer with `numpy`, `scipy`, `scikit-learn` and
   `matplotlib` (`pip install -r requirements.txt`).
   - The embedding step runs on one GPU with its own environment (`embedding/requirements.txt`).
   - The clustering reruns ran in the environment recorded in
     `datasets_merged/output/piatlas_technique_labels_v4/paper_rerun/requirements_rerun.txt`.
3. Every script uses paths relative to its own folder, so run each script from the folder it is
   in.

## Layout

| Folder | Contents |
|---|---|
| `paper_draft/PIAtlas_final/` | The taxonomy listing and its JSON export; the crosswalk data and generator; the generated LaTeX of Table VI, Table IX and Appendix F; Fig. 1 |
| `datasets_merged/output/piatlas_technique_labels_v4/` | Labeling prompts, labeling runs and calibration; the labeled templates; the verification sample (`human_annotation/`); analysis and table scripts (`paper_rerun/`) |
| `datasets_merged/output/` | `prompt_injections_whole_only_false.json`, the template-separable records the labeling run reads |
| `embedding/` | The three encoders' embedding runs, the 2-D map, the reducer comparison |

## Paper items and the files behind them

Paths below are relative to `datasets_merged/output/piatlas_technique_labels_v4/` (called `L/`)
or to `paper_draft/PIAtlas_final/` (called `P/`).

| Paper item | Files | How to regenerate |
|---|---|---|
| PIAtlas specification (App. G) | `P/artifact/piatlas_taxonomy.json`, `P/taxonomy_listing.tex` | `cd P && python3 tools/export_taxonomy_json.py` |
| Labeling prompts (§III-C) | `L/system_prompt_v1.0.txt`, `L/system_prompt_v1.1.txt`; `L/run_config.json` holds the model, the CLI flags and each prompt's SHA-256 | `cd L && python3 build_prompt.py` rebuilds v1.1 byte for byte from `L/prompt_sources/` and the listing |
| Labeling runs | `L/run_labeling.py`; progress logs `L/full*.log`; raw model output of every call in `L/calls*.jsonl.gz` (session identifiers removed) | `python3 run_labeling.py full --set wof\|curated_rest\|llmail\|f6_recheck`, then `export`. This needs the Claude Code CLI; the model is closed and not deterministic, so a rerun gives new labels |
| The 8,663 labeled templates | `L/piatlas_v4_curated_4663.json` (4,663 curated templates) and `L/piatlas_v4_llmail_4000.json.gz` (the 4,000-template LLMail-Inject sample): technique codes per template, with the class given by the code prefix | `export` builds both files from the raw label files |
| The quote behind each label | `L/labels.jsonl`, `L/labels_curated_rest.jsonl`, `L/labels_f6_recheck.jsonl`, `L/labels_llmail.jsonl`: every technique code with its evidence quote and whether the quote occurs verbatim | `L/paper_rerun/quotes_and_calibration.py` checks the quotes and that the two released files are exactly the export of these |
| Batch-size calibration (Table III, Table XX) | `L/calib/` (`b1a`, `b1b`, `b4`, `b8`, `sample.json`, `decision.json`) | `run_labeling.py calibrate`; `L/paper_rerun/quotes_and_calibration.py` recomputes the table cells offline |
| Human verification sample (App. D) | `L/human_annotation/`: `sheet_author_A.csv` (the first author's 300 annotations), `sheet_human_reference.csv` and `labels_first_author.jsonl` (the same reference in release formats), `sample.json`, `agreement.json`, `inclusion_probabilities.json`, `CODEBOOK.md`, and `README.md`. The retained `labels_quality.jsonl` and compressed raw calls document the earlier LLM quality pass but are not a second human annotation. | `python3 make_sample.py` redraws the identical sample; `python3 score_agreement.py` recomputes production-versus-human results. `materialize_first_author.py` verifies and materializes the author-certified equality of the first-author and quality-pass label sets. |
| Composition depth, Findings 1–4, class shares | `L/paper_rerun/label_numbers.py` | `cd L/paper_rerun && python3 label_numbers.py ..` prints each number with its recomputed value |
| Benchmark-by-class matrix (Table VI) | `L/paper_rerun/bench_matrix.json` (machine-readable), `bench_matrix_rows.tex`, `P/sections/tab_benchmark_matrix.tex` | `python3 gen_bench_matrix.py && python3 splice_bench_matrix.py` |
| LLMail-Inject versus curated (Finding 4) | `L/paper_rerun/llmail_stats.json` | `python3 llmail_stats.py` |
| Cluster recovery (Tables VII and XIII), five seeds, no-ASB ranking | `L/paper_rerun/rerun_paper_experiments.py`, `rerun_no_asb.py`; cached cluster assignments in `cluster_cache/`; results in `rerun_k33.json` and `rerun_no_asb_k33.json` | `K=33 python3 rerun_paper_experiments.py`, then `K=33 python3 rerun_no_asb.py`. The embeddings go in `paper_text_emb/` as `f2llm_plain_noll.npy`, `f2llm_steer_noll.npy` and `qwen_noll.npy`, copied from `embedding/out_*_noll/emb_full.npy` (or set `EMB_DIR`); with the shipped `cluster_cache/`, the clusterings are not recomputed |
| Grouping by benchmark as a baseline (App. C, Finding G1) | `L/paper_rerun/benchmark_baseline.py` | reads `cluster_cache/` |
| Map and cluster composition (Fig. 1, Table VIII) | `L/paper_rerun/fig_map_construction.py`, `map_composition.py`; map coordinates and clusters in `L/paper_rerun/map/` | `python3 fig_map_construction.py out.pdf`; the coordinates come from `embedding/regen_map.py` |
| Cluster pairs (Table XIV, Finding G3) | `L/paper_rerun/cluster_pairs.py` → `map/cluster_pairs.json` | `python3 cluster_pairs.py` |
| ASB wrappers and tools (Finding G3) | `L/paper_rerun/asb_meta.py` | `python3 asb_meta.py` |
| Reducer comparison (App. C) | `embedding/regen_dr.py` → `L/paper_rerun/map/dr_sweep_labels.npz`, scored by `map_composition.py` | `cd embedding && python3 regen_dr.py` (needs the steered embeddings) |
| Embeddings (F2LLM-v2-14B plain and steered, Qwen3-Embedding-8B) | `embedding/embed_full_notrunc.py`, `run_embed.sh`, `select_curated.sh` (keeps the 4,663 curated rows via `keep_idx_noll.json`); logs of the original runs | `sh run_embed.sh`, then `sh select_curated.sh` |
| Qwen3 HDBSCAN NWCD and silhouette (Table XIII) | `embedding/compute_full.py` → `full_results.json`, `ndim_table.md` | — |
| Crosswalk of prior taxonomies (Table IX, App. F) | `P/tools/crosswalk_data.json` (one card per scheme, machine-readable), `P/tools/crosswalk_stats.json` (computed marks and counts) | `cd P && python3 tools/build_crosswalk.py` rewrites `sections/app_f_crosswalk.tex` and `sections/crosswalk_counts.tex` byte for byte |
| Category names against the sources | `P/tools/check_crosswalk_sources.py` | needs local copies of the cited works under `literature/`; they are not redistributed |

## Method details

- **Labeling.** Claude Opus 5.5 with medium effort ran through Claude Code 2.1.281 in headless
  mode, with no tools and no MCP servers, one record per call (batch size 1, chosen by the
  calibration rule in `run_labeling.py`), in a fixed seeded order.
  - The first pass used prompt v1.0.
  - Prompt v1.1 adds one rule: a general request to answer carefully or in detail is not F6. The
    478 curated templates whose first labels carried F6 were relabeled with v1.1
    (`labels_f6_recheck.jsonl`), and the 4,000 LLMail-Inject templates were labeled with v1.1.
  - The 64 records of calibration run `b1a` keep their calibration labels in `labels.jsonl`.
  - One ASB record (position 1178) whose first answer was `cannot_analyze` was relabeled once
    (run `retry_nonok` in `labels.jsonl`).
- **Human verification.** One human annotator, the first author, labeled all 300 shuffled
  verification records at the class and technique levels while blind to the production labels and
  source dataset. There was no second human annotator, cross-verification, adjudication, or
  human--human inter-rater reliability. The agreement statistics compare the production labels
  with the first-author reference. See `L/human_annotation/README.md` for the complete file map and
  protocol.
- **Clustering.** k-means is `MiniBatchKMeans` on L2-normalized embeddings (batch 1024, k=33, the
  number of techniques that occur in the labels). The five-seed runs use seeds 0, 1, 2, 3 and 42
  with 10 initializations each.
  - The Gaussian mixture uses diagonal covariances.
  - HDBSCAN uses `min_cluster_size=15`, and its noise points count as one cluster.
- **Map.** The map is PCA to 50 dimensions, then UMAP (15 neighbors, `min_dist` 0, seed 42),
  then k-means with k chosen by silhouette over 8–40.
  - The paper's map (k=10) reproduces with the package versions in `embedding/requirements.txt`.
  - The map rebuilt inside `rerun_paper_experiments.py` under newer versions differs, so use
    `embedding/regen_map.py`.

## License

The code is released under the MIT license (`LICENSE`). The labeled templates are drawn from
public datasets whose licenses are listed in Appendix A of the paper.
