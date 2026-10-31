# Assignment 3 shared worklog

## Current session: review and local commits

- Existing staged files match the preceding A3 work; no unrelated changes found.
  Configured Git identity is Trym Johnsen's personal identity and is retained.
- Reviewed the whole task PDF, worklog, source dependencies, staged diff and file
  sizes. `git diff --cached --check` and `src/checks.py` pass.
- Preserve the initial implementation and its results as one coherent commit:
  splitting its cross-imported scripts, evidence and handover would create
  artificial incomplete states. New quality review and model comparison will be
  separate steps. No raw recordings, window parquet or weights are added.
- Next: inspect measured/interpolated/uncovered support before defining the
  bounded comparison plan. Do not start ensembles or deep learning this session.

This is the main handover for Eivind and Trym. Status words mean:
**planned** = not built; **implemented** = code exists; **run** = command completed
and has recorded output. A result is not evidence until its run is recorded here.

## Problem and scope — agreed

Can eye-tracking data identify which of two known code tasks, `vehicle` or
`rectangle`, a new participant is viewing? We study what sharing eye-tracking
measurements can reveal about known screen content when the recipient receives
neither task names nor screenshots. Task names are used only to construct ground
truth and audit the data; models never receive them.

We use code screens only, not multiple-choice screens. We compare features with
and without absolute screen position. This is a small study of two known tasks,
not a measure of comprehension, reconstruction of source code, or evidence of
generalisation to arbitrary programs. No claim of real-time prediction is made.

## A1 reference and starting state — checked

- A1 reference commit: `8b3337370f02242ba2132a9e4b84ee2bc4669426`.
- Git status was clean before this work. No local edits needed preserving.
- No AGENTS.md was found in this repository or its ancestor directories.
- A1 has 33 recordings (32 TSV + participant 211 in Excel), at nominal 250 Hz.
  Each SMP row is one time point, normally about 4 ms after the previous one.
- A1 cache: 1,440,469 rows, including non-code screens. It precedes velocity
  removal and pupil clipping. Its timestamps and quality labels are available.
- Reuse A1 loaders, cleaning decisions/cache, physical velocity definition and
  original participant split. Keep all original code, raw data, cache, figures
  and reports in place. Shared README and ignore rules may change.
- The official A3 PDF was found at the root as
  `IT3212-2026-Assignment 3 (1).pdf`; moved to `assignment3/assignment.pdf`.
  No repository references to its old filename were found.
- `results.md` contains older A1 feature/PCA counts (34/29; eight components at
  90%). Current A1 code/report use 32/27 and seven components. Do not rewrite the
  historical log. PDF/HTML exports also predate the latest report.md edits.

## Requirement status — after the first run

| # | Requirement | Current evidence and remaining work |
|---|---|---|
| 1 | Practical and ML problem | Agreed above; formal report planned |
| 2 | Justified preprocessing | Implemented and run; raw/cache consistency and window checks pass; full report pending |
| 3 | Justified feature extraction | 13 summaries implemented, run and documented below |
| 4 | Selection/reduction or justified omission | Explicit 13/11-feature selection implemented and compared; no A3 PCA |
| 5 | Three algorithms and justification | One decision-tree algorithm run in two variants; dummy is a reference, not a second allowed algorithm; two algorithms still needed |
| 6 | Explain comparison of three | Initial dummy/ablation comparison run; three-algorithm comparison pending |
| 7 | Boosting and bagging, different base algorithms | Planned; wording open |
| 8 | Basic vs boosting vs bagging vs transfer | Basic results only; ensembles pending and transfer undefined |
| 9 | CNN or deep network with justified tuning | Planned |
| 10 | Complexity, time, performance in practical context | Initial fit/predict times and tree depth/node counts recorded; full comparison pending |

The PDF says 200 points, pass at 130. It specifies no submission format, deadline,
minimum dataset size, library restriction or requirement to implement from
scratch. Check general course rules separately. Justifications carry marks.

## Decisions before seeing scores

1. Keep A1 test IDs **2, 6, 8, 18, 25, 28, 37** locked. No model will fit, predict,
   tune or calculate scores on them in this first experiment. Dataset quality
   counts may include them. Use three participant-grouped folds on development
   people only, seed 42.
2. Use fixed two-second half-open windows anchored at raw stimulus messages.
   Never cross a person, segment or screen change. Drop incomplete final windows.
   Measure timestamp-based sample support, rather than joining rows after gaps.
   Require at least 90% supported time and at most 150 ms continuously uncovered
   inside a window. These are quality rules chosen before model results.
3. Use cache signals but recover original display boundaries from raw recordings.
   A cache alone cannot tell us the original screen onset after leading rows were
   deleted. Do not rebuild or change the A1 cache.
4. Do not apply A1's per-person pupil clipping in A3's first run. It would use the
   whole validation participant's recording. The cache is not pupil-clipped;
   inspect the retained range and keep values. A small tree does not require
   scaling. Alternative: learn common clipping bounds on fold-training data only.
5. Reuse cached blink/gap features with their exact A1 meaning: rolling **1,250
   sample rows**, nominally five seconds, grouped by stimulus before deletion.
   This is not an exact five-second clock window, nor a fresh blink count in our
   two-second window. Do not reconstruct missing blink events from cache rows.
6. Compare the same small tree on an explicit full feature list and its subset
   without absolute position. Exclude raw coordinates, eye position, gaze vectors,
   validity/confidence flags, language and all identity/label/time metadata.
   Relative velocity can still carry spatial structure. Group B is not spatially
   information-free.
7. Use a depth-four tree, minimum leaf size ten and balanced class weights, fixed
   before scores. Compare to the most-frequent-class DummyClassifier. No tuning,
   PCA, ensembles or neural networks in this first run.
8. Do not apply PCA in this first experiment: the selected 13 summaries are few,
   interpretable and a small tree does not require scaling. PCA would mix position
   and other signals, complicating our intended position comparison. The explicit
   feature selection addresses point 4; later models may justify other choices.

## Implemented/run record

- Planning and preservation checks completed. Data builder, experiment and
  targeted checks implemented in three scripts and run successfully. The dated
  run sections below contain the actual results; this list records design context.
- Pre-model cache inspection: 856,785 code samples, 31 people and 60 displays;
  572 retained timestamp intervals exceed 150 ms, maximum 25.678280 s. Treating
  these rows as a continuous 250 Hz signal would be incorrect.
- Retained, unclipped code pupil ranges inspected before scoring: left 1.87–5.33
  mm; right 1.70–4.76 mm. These do not indicate an obvious need for whole-person
  statistical clipping. This is not proof that every pupil sample is artefact-free.
- The builder also reruns A1 preprocess one raw file at a time, without writing
  A1 output, and compares code timestamps, used signals and quality labels with
  the cache. This checks consistency and recovers exclusions and screen onsets.
- Group A has 13 features; B has 11. A adds only mean screen x/y. Both retain
  x/y population standard deviations, which are translation-invariant spread,
  plus pupil means/SDs, speed median/p90, cached blink/gap means and interpolation
  fraction. B still contains relative spatial structure.
- Speed summaries use in-window consecutive samples only, excluding intervals
  over 20 ms and speeds over 1,000 deg/s. This avoids treating motion across a
  deleted hole as a short observed movement. A1's physical velocity filter is
  applied first. Pupil features remain unclipped; no fitted preprocessing occurs.
- Three folds are generated by seeded KFold on sorted development participant
  IDs, then expanded to all their windows. This gives equal numbers of people per
  fold rather than optimising fold scores. Models use identical folds.

## Data build — run, 31 October 2026

Command: `.venv/bin/python -B assignment3/src/data.py` from the repository root.
Configuration: `config.json`. Output: `cache/windows.parquet` (ignored by Git),
`results/data/summary.json`, `recordings.csv`, `displays.csv`, `window_audit.csv`
and `provenance.json`. No model was evaluated during construction.

- All 33 raw recordings were read: 1,710,878 SMP measurements and 377 messages.
- There are 66 raw code displays. A1's 20-second rule retains 60, from 31 people.
  All 60 were checked against rerun A1 preprocessing: timestamps, six used signal
  columns and quality labels agree. Root cache and other A1 outputs were not rebuilt.
- Code-only cache: 856,785 rows. A1 physical velocity rule removes 138 code rows,
  leaving 856,647: 510,927 vehicle / 345,720 rectangle. At nominal 250 Hz this is
  3,426.588 sample-equivalent seconds, not a continuous recording duration.
- Of 1,819 full two-second candidate windows, 1,366 pass and 453 fail. Reasons:
  277 low coverage plus a long uncovered interval; 175 long interval alone;
  one low coverage alone. See every candidate, including rejected ones, in the
  window audit. Partial windows at display ends are not candidates.
- Final table: **30 people, 59 displays, 1,366 windows** (790 vehicle / 576
  rectangle), spanning 2,732 seconds of windows. Conservative timestamp support
  is **2,724.721542 s**, including interpolated samples, not only direct observations.
- Development: **23 people, 992 windows**, 562 vehicle / 430 rectangle;
  1,978.406376 supported seconds inside 1,984 window seconds.
- Locked test: **7 people, 374 windows**, 228 vehicle / 146 rectangle;
  746.315166 supported seconds inside 748 window seconds. These are dataset
  counts only; no test predictions or scores have been computed.
- Participant 1's surviving vehicle display produces no accepted windows under
  the predeclared rule. This reduces development from 24 to 23 people. We did not
  relax the quality rule or use scores to decide exclusion. Participant 18's
  rectangle display was already removed by A1, so test data contain 13 displays.

Feature definitions: sample-row means and population standard deviations
(`ddof=0`) within each accepted window. Velocity is displacement divided by real
elapsed time and A1's pixels-per-degree constant; summarised by median and p90.
Only consecutive pairs within that window and at most 20 ms apart contribute.
The cached A1 blink-rate and gap-fraction signals are averaged, not reinterpreted
as counts. Interpolation fraction is interpolated rows divided by retained rows
in the window. Uneven sampling is bounded by quality checks, but these summaries
are row-weighted, not time-weighted. Full explicit lists are in data.py and the
data summary; only these names may be supplied to models.

## Targeted checks — run, before model validation

Command: `.venv/bin/python -B assignment3/src/checks.py`.
Output: `results/data/checks.json`, **PASS**.
Checks cover all 1,819 candidates, 1,366 accepted windows, explicit feature
allowlists, person separation, original screen boundaries, non-overlap, coverage,
and synthetic empty windows, deleted 200 ms holes and half-open boundaries.
54 protected A1 files, including raw data, code, figures, reports and cache,
match their recorded SHA-256 hashes.

Further data diagnostics, not score-based exclusions:

- A1-excluded displays have usable sample-equivalent seconds: pid 1 rectangle
  13.572; pid 18 rectangle 5.028; pid 22 rectangle 2.352 and vehicle 1.148;
  pid 39 vehicle 15.488 and rectangle 6.652. All are below 20 s.
- Retained displays span 3,704.004562 clock seconds. Full candidates span 3,638
  seconds; incomplete trailing remainders sum to 66.004562 s.
- Accepted coverage: minimum 90.19005%, median 99.87475%; largest continuous
  uncovered interval 142.359 ms.
- Accepted windows average 3.5066% interpolated rows, but one has **76%**.
  Coverage includes A1-interpolated support, so high coverage alone does not mean
  mostly direct measurements. This limitation remains visible in the features
  and audit, without adding an unplanned exclusion rule.

## First validation — run, 31 October 2026

Command: `.venv/bin/python -B assignment3/src/experiment.py`.
Output directory: `results/first_validation/`. See `config.json`, `folds.json`,
`fold_metrics.csv`, `participant_metrics.csv`, `predictions.csv`, `summary.json`
and `environment.json`. The exact installed package versions are also saved in
`requirements.txt` in that directory. Model source and window-table hashes are
recorded. No fitted model is saved, and no locked test person is scored.

The seed-42 participant split gives validation folds of 8, 8 and 7 people (315,
359 and 318 windows). All their windows stay together. Every development person
appears in validation exactly once, and every one has both classes after window
selection. All three model variants use identical folds. There are no learned
transforms: no scaler, imputer, PCA or pupil clipping. Each classifier fits only
the current fold's training windows.

Results below are **actual development validation**, not final-test performance.
Mean and sample SD are across the three folds; they are not confidence intervals.

| Model | Balanced accuracy, mean ± SD | Macro-F1, mean ± SD | Equal-person balanced accuracy | Equal-person macro-F1 |
|---|---|---|---|---|
| Most-frequent dummy | 0.5000 ± 0.0000 | 0.3621 ± 0.0182 | 0.5000 | 0.3557 |
| Tree A, with mean position | 0.6114 ± 0.0233 | 0.5930 ± 0.0373 | 0.6191 | 0.5789 |
| Tree B, without absolute position | 0.5072 ± 0.0159 | 0.4960 ± 0.0240 | 0.5207 | 0.4700 |

Fold balanced accuracies: A **0.6161, 0.5860, 0.6319**; B **0.5051, 0.5241,
0.4924**. B is below the dummy on balanced accuracy in the last fold. We keep this
negative result. Macro-F1 can improve over a majority dummy even while balanced
accuracy remains near 0.5; the two metrics answer different questions.

Per-person CSV gives both scores and class counts for all 23 people and each
model. Equal-person means weight each person once, irrespective of their number
of windows. Pooled out-of-fold scores in summary.json instead weight windows.
The code defines per-person macro-F1 over observed true classes and records
class support; a future one-class person would need cautious interpretation.
All people in this actual development run have both classes.

| Model | Total fit time, 3 folds | Total predict time, 3 folds | Tree depth / node counts |
|---|---|---|---|
| Dummy | 0.06224 s | 0.000934 s | Not a tree |
| Tree A | 0.03247 s | 0.002099 s | Depth 4; 15, 19, 17 nodes |
| Tree B | 0.02783 s | 0.002048 s | Depth 4; 23, 19, 19 nodes |

Timing uses `perf_counter` wall time around a single fit and a single predict,
with input arrays prepared first. It excludes raw loading, window construction,
I/O and metric calculations. No repeats or warm-up were used; the first dummy
fit took 0.06096 s while its later fits were under 0.001 s. Initial overhead and
very small prediction calls make these timings noisy. Do not conclude that the
dummy is computationally more complex than the tree from these numbers.

**Interpretation:** mean screen position helps this particular small tree on this
development sample: removing only mean x/y reduces fold-average balanced accuracy
by about 0.104. A's performance is modest, not reliable identification of every
window or every person. B is near the dummy on balanced accuracy. This is initial
evidence about these two known tasks and this feature/model choice, not proof
that removing position makes shared data private. Relative spread, velocity and
other channels can still carry task information; other models may use it better.

An independent check recomputed fold metrics from the saved 2x2 confusion matrices,
checked all 2,976 prediction rows and the 69 participant-metric rows, verified no
test predictions and one validation fold per development person, and compared
pre-edit A1 hashes. Result: `results/first_validation/checks.json`, **PASS**.
The moved A3 PDF also matches its original Git blob byte for byte.

## Open questions and limits

- What does **transfer** in point 8 require? It appears without an implementation
  instruction. It is neither assumed optional nor claimed complete.
- Does point 7 require distinct base learner families for boosting and bagging?
  Our working reading is yes; confirm before selecting them.
- Can the point-9 network count as one of point 5's three algorithms? Unclear.
- A1 linear interpolation uses later samples. Offline results cannot establish
  real-time prediction performance.
- Two fixed tasks confound task identity with layout. Only one person saw Scala.
  Correlated windows are not independent participants.
- The A1 cache's cleaning provenance must be preserved and audited; this first
  stage is not a full independent revalidation of every A1 cleaning rule.
- The quality rule removes participant 1 completely and some windows from other
  people. Scores describe the retained data, not every captured sample.
- Interpolated support can dominate individual accepted windows; investigate this
  before treating coverage as a guarantee of measurement quality.
- Three development folds and 23 people do not establish a population-wide
  privacy risk estimate. The final test is still untouched.
- This supervised experiment assumes labelled reference recordings of the known
  tasks. Window construction also uses producer-side screen-change boundaries.
  It does not demonstrate identification from an arbitrary unsegmented stream
  with no labelled reference data. Task names/images are not model inputs.

## Next work and handover — planned

Eivind: start with this log, README.md and the two summary.json files. Inspect
participant_metrics.csv and the window audit, especially participant 1 and the
window with 76% interpolation. Then review the fixed rules with Trym. The initial
build/experiment/checks are complete; do not repeat the A1 pipeline to start A3.

Mandatory next steps, in order:

1. Confirm the transfer and base-algorithm wording with the course staff. Decide
   whether the data quality policy needs a reasoned revision before more models.
2. Add one bounded kNN experiment on the same development folds and feature groups,
   with scaling fitted within training folds only. Then add kernel SVM to reach
   three allowed algorithms. Keep searches small and document choices.
3. Implement the required ensembles after agreeing their base learners; implement
   and tune a small deep model; compare complexity, time and performance fairly.
4. Freeze model choices, evaluate the locked test once, and complete the report.

Optional improvements after mandatory work is under control: a predeclared
interpolation/coverage sensitivity analysis, better clock-based blink features,
additional participants/tasks and more stable timing measurements.

Eivind may reconsider feature definitions, quality thresholds, pupil treatment
and eventual model choices. Record the reason and new configuration before
looking at new validation scores. Never choose exclusions or the question for
a desired score, and never use the locked test to guide these choices.
