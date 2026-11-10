# PIAtlas and PICorpus: code and data

Code and data for the paper **SoK: PIAtlas: Decomposing the Unbounded Prompt Injection Attack Space**.

[Paper (arXiv:2610.24999)](https://arxiv.org/abs/2610.24999) · [Interactive taxonomy](https://stevendamianakis.github.io/piatlas-site/) · [Cite](#citation)

PIAtlas is a taxonomy of prompt-injection attacks, organized by how the injected instruction is
constructed to subvert the model. It has 6 classes, 34 techniques and 131 sub-techniques.

**Browse the taxonomy on the website: <https://stevendamianakis.github.io/piatlas-site/>**

Every entry there has a definition and a worked example, and the whole taxonomy can be downloaded
as CSV, JSON or Markdown.

## What is in this repository

- The PIAtlas specification
- The labeling prompts and runs
- The 8,663 construction-labeled templates
- The calibration and verification data
- The crosswalk of prior taxonomies
- The scripts that regenerate the paper's tables and figure

## Setup

1. Unpack the data once, from the repository root. Two JSON inputs are stored as `.json.gz`, and
   the script unpacks them in place.

   ```sh
   sh unpack_data.sh
   ```

2. Install the packages the analysis scripts need: Python 3.12 or newer with `numpy`, `scipy`,
   `scikit-learn` and `matplotlib`.

   ```sh
   pip install -r requirements.txt
   ```

3. Run each script from the folder it is in. Every script uses paths relative to its own folder.

Two steps have their own environments:

- The embedding step runs on one GPU with its own environment (`embedding/requirements.txt`).
- The clustering reruns ran in the environment recorded in
  `datasets_merged/output/piatlas_technique_labels_v4/paper_rerun/requirements_rerun.txt`.

## Layout

| Folder | Contents |
|---|---|
| `paper_draft/PIAtlas_final/` | The taxonomy listing and its JSON export; the crosswalk data and generator; the generated LaTeX of Table VI, Table IX and Appendix F; Fig. 1 |
| `datasets_merged/output/piatlas_technique_labels_v4/` | Labeling prompts, labeling runs and calibration; the labeled templates; the verification sample (`human_annotation/`); analysis and table scripts (`paper_rerun/`) |
| `datasets_merged/output/` | `prompt_injections_whole_only_false.json`, the template-separable records the labeling run reads |
| `embedding/` | The three encoders' embedding runs, the 2-D map, the reducer comparison |

## Paper items and the files behind them

Each group lists the files behind a paper item and how to regenerate it. Select a group to open it.

Paths are relative to one of two folders:

- `L/` is `datasets_merged/output/piatlas_technique_labels_v4/`
- `P/` is `paper_draft/PIAtlas_final/`

<details>
<summary><b>Taxonomy and crosswalk</b> · App. G, Table IX, App. F</summary>

**PIAtlas specification** (App. G)

- Files: `P/artifact/piatlas_taxonomy.json`, `P/taxonomy_listing.tex`
- Regenerate: `cd P && python3 tools/export_taxonomy_json.py`

**Crosswalk of prior taxonomies** (Table IX, App. F)

- Files: `P/tools/crosswalk_data.json` (one card per scheme, machine-readable),
  `P/tools/crosswalk_stats.json` (computed marks and counts)
- Regenerate: `cd P && python3 tools/build_crosswalk.py` rewrites
  `sections/app_f_crosswalk.tex` and `sections/crosswalk_counts.tex` byte for byte

**Category names against the sources**

- Files: `P/tools/check_crosswalk_sources.py`
- Regenerate: needs local copies of the cited works under `literature/`; they are not
  redistributed

</details>

<details>
<summary><b>Labeling</b> · §III-C, Table III, Table XX</summary>

**Labeling prompts** (§III-C)

- Files: `L/system_prompt_v1.0.txt`, `L/system_prompt_v1.1.txt`. `L/run_config.json` holds the
  model, the CLI flags and each prompt's SHA-256
- Regenerate: `cd L && python3 build_prompt.py` rebuilds v1.1 byte for byte from
  `L/prompt_sources/` and the listing

**Labeling runs**

- Files: `L/run_labeling.py`; progress logs `L/full*.log`; raw model output of every call in
  `L/calls*.jsonl.gz` (session identifiers removed)
- Regenerate: `python3 run_labeling.py full --set wof|curated_rest|llmail|f6_recheck`, then
  `export`
- Note: this needs the Claude Code CLI. The model is closed and not deterministic, so a rerun
  gives new labels

**The 8,663 labeled templates**

- Files: `L/piatlas_v4_curated_4663.json` (4,663 curated templates) and
  `L/piatlas_v4_llmail_4000.json.gz` (the 4,000-template LLMail-Inject sample). Each holds the
  technique codes per template, with the class given by the code prefix
- Regenerate: `export` builds both files from the raw label files

**The quote behind each label**

- Files: `L/labels.jsonl`, `L/labels_curated_rest.jsonl`, `L/labels_f6_recheck.jsonl`,
  `L/labels_llmail.jsonl`. Each holds every technique code with its evidence quote and whether
  the quote occurs verbatim
- Regenerate: `L/paper_rerun/quotes_and_calibration.py` checks the quotes and that the two
  released files are exactly the export of these

**Batch-size calibration** (Table III, Table XX)

- Files: `L/calib/` (`b1a`, `b1b`, `b4`, `b8`, `sample.json`, `decision.json`)
- Regenerate: `run_labeling.py calibrate`; `L/paper_rerun/quotes_and_calibration.py` recomputes
  the table cells offline

</details>

<details>
<summary><b>Human verification</b> · App. D</summary>

**Human verification sample** (App. D)

- Files, all in `L/human_annotation/`:
  - `sheet_author_A.csv`: the first author's 300 annotations
  - `sheet_human_reference.csv` and `labels_first_author.jsonl`: the same reference in release
    formats
  - `sample.json`, `agreement.json`, `inclusion_probabilities.json`, `CODEBOOK.md` and
    `README.md`
  - The retained `labels_quality.jsonl` and compressed raw calls document the earlier LLM
    quality pass but are not a second human annotation
- Regenerate:
  - `python3 make_sample.py` redraws the identical sample
  - `python3 score_agreement.py` recomputes production-versus-human results
  - `materialize_first_author.py` verifies and materializes the author-certified equality of
    the first-author and quality-pass label sets

</details>

<details>
<summary><b>Label analysis and tables</b> · Findings 1–4, Table VI</summary>

**Composition depth, Findings 1–4, class shares**

- Files: `L/paper_rerun/label_numbers.py`
- Regenerate: `cd L/paper_rerun && python3 label_numbers.py ..` prints each number with its
  recomputed value

**Benchmark-by-class matrix** (Table VI)

- Files: `L/paper_rerun/bench_matrix.json` (machine-readable), `bench_matrix_rows.tex`,
  `P/sections/tab_benchmark_matrix.tex`
- Regenerate: `python3 gen_bench_matrix.py && python3 splice_bench_matrix.py`

**LLMail-Inject versus curated** (Finding 4)

- Files: `L/paper_rerun/llmail_stats.json`
- Regenerate: `python3 llmail_stats.py`

</details>

<details>
<summary><b>Embeddings, clustering and map</b> · Fig. 1, Tables VII, VIII, XIII and XIV, App. C</summary>

**Embeddings** (F2LLM-v2-14B plain and steered, Qwen3-Embedding-8B)

- Files: `embedding/embed_full_notrunc.py`, `run_embed.sh`, `select_curated.sh` (keeps the
  4,663 curated rows via `keep_idx_noll.json`); logs of the original runs
- Regenerate: `sh run_embed.sh`, then `sh select_curated.sh`

**Cluster recovery** (Tables VII and XIII), five seeds, no-ASB ranking

- Files: `L/paper_rerun/rerun_paper_experiments.py`, `rerun_no_asb.py`; cached cluster
  assignments in `cluster_cache/`; results in `rerun_k33.json` and `rerun_no_asb_k33.json`
- Regenerate: `K=33 python3 rerun_paper_experiments.py`, then `K=33 python3 rerun_no_asb.py`
- Note: the embeddings go in `paper_text_emb/` as `f2llm_plain_noll.npy`,
  `f2llm_steer_noll.npy` and `qwen_noll.npy`, copied from `embedding/out_*_noll/emb_full.npy`
  (or set `EMB_DIR`). With the shipped `cluster_cache/`, the clusterings are not recomputed

**Qwen3 HDBSCAN NWCD and silhouette** (Table XIII)

- Files: `embedding/compute_full.py` → `full_results.json`, `ndim_table.md`

**Grouping by benchmark as a baseline** (App. C, Finding G1)

- Files: `L/paper_rerun/benchmark_baseline.py`
- Regenerate: reads `cluster_cache/`

**Map and cluster composition** (Fig. 1, Table VIII)

- Files: `L/paper_rerun/fig_map_construction.py`, `map_composition.py`; map coordinates and
  clusters in `L/paper_rerun/map/`
- Regenerate: `python3 fig_map_construction.py out.pdf`; the coordinates come from
  `embedding/regen_map.py`

**Cluster pairs** (Table XIV, Finding G3)

- Files: `L/paper_rerun/cluster_pairs.py` → `map/cluster_pairs.json`
- Regenerate: `python3 cluster_pairs.py`

**ASB wrappers and tools** (Finding G3)

- Files: `L/paper_rerun/asb_meta.py`
- Regenerate: `python3 asb_meta.py`

**Reducer comparison** (App. C)

- Files: `embedding/regen_dr.py` → `L/paper_rerun/map/dr_sweep_labels.npz`, scored by
  `map_composition.py`
- Regenerate: `cd embedding && python3 regen_dr.py` (needs the steered embeddings)

</details>

## Method details

### Labeling

- Claude Opus 5.5 with medium effort ran through Claude Code 2.1.281 in headless mode, with no
  tools and no MCP servers.
- It labeled one record per call (batch size 1, chosen by the calibration rule in
  `run_labeling.py`), in a fixed seeded order.
- The first pass used prompt v1.0.
- Prompt v1.1 adds one rule: a general request to answer carefully or in detail is not F6. The
  478 curated templates whose first labels carried F6 were relabeled with v1.1
  (`labels_f6_recheck.jsonl`), and the 4,000 LLMail-Inject templates were labeled with v1.1.
- The 64 records of calibration run `b1a` keep their calibration labels in `labels.jsonl`.
- One ASB record (position 1178) whose first answer was `cannot_analyze` was relabeled once
  (run `retry_nonok` in `labels.jsonl`).

### Human verification

- One human annotator, the first author, labeled all 300 shuffled verification records at the
  class and technique levels while blind to the production labels and source dataset.
- There was no second human annotator, cross-verification, adjudication, or human-human
  inter-rater reliability.
- The agreement statistics compare the production labels with the first-author reference.
- See `L/human_annotation/README.md` for the complete file map and protocol.

### Clustering

- k-means is `MiniBatchKMeans` on L2-normalized embeddings (batch 1024, k=33, the number of
  techniques that occur in the labels).
- The five-seed runs use seeds 0, 1, 2, 3 and 42 with 10 initializations each.
- The Gaussian mixture uses diagonal covariances.
- HDBSCAN uses `min_cluster_size=15`, and its noise points count as one cluster.

### Map

- The map is PCA to 50 dimensions, then UMAP (15 neighbors, `min_dist` 0, seed 42), then k-means
  with k chosen by silhouette over 8–40.
- The paper's map (k=10) reproduces with the package versions in `embedding/requirements.txt`.
- The map rebuilt inside `rerun_paper_experiments.py` under newer versions differs, so use
  `embedding/regen_map.py`.

## Citation

```bibtex
@misc{damianakis2026piatlas,
  title         = {{SoK}: {PIAtlas}: Decomposing the Unbounded Prompt Injection Attack Space},
  author        = {Damianakis, Stavros and Psounis, Konstantinos},
  year          = {2026},
  eprint        = {2610.24999},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2610.24999}
}
```

## License

The code is released under the MIT license (`LICENSE`). The labeled templates are drawn from
public datasets whose licenses are listed in Appendix A of the paper.
