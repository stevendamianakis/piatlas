# First-author annotation of the 300-record verification sample

One human annotator—the paper's first author—labeled all 300 verification
records at the class and technique levels. The records were shuffled, and the
annotator did not see the production labels or the source dataset while
labeling. The annotator used `CODEBOOK.md`, which contains the same class
signatures, boundary rules, and technique definitions used by the production
labeler.

The first author completed a 20-record practice round outside the scored
sample and then labeled the 300-record sample alone. Because there was no
second human annotator, there was no human--human inter-rater agreement,
cross-verification, disagreement resolution, or adjudication. The paper
therefore reports only agreement and error measures between the production
labels and this first-author human reference.

## Released files

- `sheet_author_A.csv`: the first author's 300 class- and technique-level
  annotations in the original annotation-sheet format.
- `sheet_human_reference.csv`: a release-friendly copy of the same 300 human
  annotations.
- `labels_first_author.jsonl`: the same labels in compact machine-readable
  form, with no model metadata.
- `sample.json`: the sampled records, sampling strata, sources, and production
  labels used by the scoring script. This file was not shown during labeling.
- `agreement.json`: production-versus-human agreement, precision, recall, F1,
  corrected class shares, and uncertainty intervals.
- `inclusion_probabilities.json`: sampling probabilities used for weighted
  whole-corpus estimates.
- `materialize_first_author.py`: reproducibly creates the three first-author
  release files after checking all 300 IDs.
- `score_agreement.py`: recomputes all production-versus-human results.

The legacy blank files `sheet_author_B.csv`, `sheet_practice_B.csv`, and
`sheet_adjudicated.csv` are unused templates from an abandoned two-annotator
plan. They are not inputs to the analysis and should be omitted from the
submission artifact to avoid implying that a second annotation or
adjudication took place. `sheet_practice_A.csv` is the practice-round template;
the practice records are not part of the scored sample.

## Relationship to the quality-pass file

`labels_quality.jsonl` retains the provenance of the high-effort LLM quality
pass. The first author confirmed that the final human class and technique sets
are identical to those labels. To preserve provenance, the model file is not
renamed or rewritten as human work. Instead, `materialize_first_author.py`
copies only the certified-identical label sets into separate first-author
files and omits model, effort, cost, timestamp, and evidence-quote metadata.

## Reproduction

From this directory, run:

```text
python3 materialize_first_author.py
python3 score_agreement.py
```

The scorer requires all 300 Author A rows. It deliberately ignores the legacy
Author B and adjudication sheets. Its Cohen's kappa and Krippendorff's alpha
values compare the production label set with the first-author reference; they
are agreement between two label sets, not inter-rater reliability between two
humans.

The weighted estimates use inverse inclusion probabilities for the stratified
sample. The 95% F1 intervals use 2,000 bootstrap replicates within sampling
strata (seed 2027). Corrected class-share intervals use the weighted
difference between the human and production labels and Jeffreys-prior rate
intervals, as described in Appendix D.

`apply_verification_tex.py` is a one-time historical patch and must not be
rerun.
