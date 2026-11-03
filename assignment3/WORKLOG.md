# Assignment 3 shared worklog

Main handover for Eivind and Trym. **Done** means executed and checked for this
analysis, not that the final submission is complete. Planned work has no results.
Detailed initial notes remain in Git: `git show 2be417f:assignment3/WORKLOG.md`.

## Current handover

- Trym (to `4656945`): windows, V1/V2 quality, nested tree/kNN/RBF-SVM on A/B.
  All V1/V2 result files remain byte-for-byte unchanged.
- Eivind (branch `eivind/assignment3`, from 2026-11-02): problem reframed around
  A vs B and gaze-data privacy; V2 reproduced on macOS; V3 I-VT event features;
  mutual-information selection; AdaBoost and bagged kNN; sequence CNN with
  transfer; one-time locked-test evaluation; figures and report.md. Each step's
  plan was committed before its scores; amendments are dated in the plan files.
- Start with report.md, then the summary.csv in each results/ subdirectory.
- Open: ask course staff what "transfer" means and about format/deadline;
  Trym should confirm the reframing, his contribution text and the AI declaration;
  personal reflections, if required, are for each author to write.

## Problem and scope

Real-world problem: eye trackers are entering consumer devices, and a common
safeguard is to share derived gaze signals (pupil size, blinks, eye speed)
instead of raw screen coordinates. Is that enough to hide what someone reads?

ML problem: binary classification of two-second gaze windows, vehicle or
rectangle, for previously unseen participants, scored by balanced accuracy with
equal weight per person. Feature group A includes mean gaze position; B removes
it. The A–B difference shows how much information lies in where people look
rather than how they look. Use code screens only; task names and screenshots are
never model inputs.

This is a closed-set, offline study with two known texts; not comprehension
measurement, source reconstruction, arbitrary-program recognition, evidence of
causation, or a privacy guarantee. A weak B result does not show that content
cannot leak through other signals, models or larger datasets.

The supervised experiment assumes labelled reference recordings and producer-side
screen boundaries. It does not demonstrate an attack on an arbitrary unsegmented
stream with no reference labels. Linear interpolation uses later samples.

Framing revised 2026-11-02 (Eivind), after reading the V2 results: A/B was
already in the plan committed before scores (`2c2edf8`), but making it the main
question and adding the privacy motivation came afterwards. No data, code, plan
or results changed.

## Requirement status

| # | Requirement | Status and evidence |
|---|---|---|
| 1 | Practical and ML problem | Done: privacy framing and A/B question above (revised 2026-11-02); final report still needed |
| 2 | Justified preprocessing | Done for this analysis: A1 reuse and V2 time-quality rule |
| 3 | Justified feature extraction | Done: 13 window summaries, plus 8 V3 I-VT event features |
| 4 | Selection/reduction or justified omission | Done: A/B design, no PCA, and V3 mutual-information selection inside CV |
| 5 | Three algorithms and justification | Done: tree, kNN and kernel SVM, equal bounded search budgets |
| 6 | Compare/explain three algorithms | Done for development: nested OOF comparison and class/person errors |
| 7 | Boosting/bagging, different base algorithms | Done for development: AdaBoost (trees) and bagged kNN on A3/B3 |
| 8 | Basic vs boosting vs bagging vs transfer | Done for development; transfer = CNN pretraining on other screens (ask staff) |
| 9 | CNN/deep network with justified tuning | Done for development: 1D CNN, nested tuning, seed spread |
| 10 | Complexity, time, performance in context | Done: sizes and timings for every method, locked test run once |

The one-page PDF says 200 points, pass at 130. No deadline, submission format,
minimum sample size or library restriction is stated there; check course-wide
rules. Justifications carry marks. The final report is not yet written.

## Data and A1 preservation

- A1 reference: `8b3337370f02242ba2132a9e4b84ee2bc4669426`.
- 33 raw recordings: 32 TSV + Excel participant 211; nominal 250 Hz. One SMP row
  is one time point, normally about 4 ms apart. Raw data are tracked in Git.
- Preserve A1 code/data/cache/figures/reports at root. Only the shared README and
  ignore rules changed. Moved the A3 PDF to `assignment3/assignment.pdf` unchanged.
- No AGENTS.md found. The initial staged changes matched preceding A3 work, with
  no unrelated edits. Cross-imported source, evidence and docs were committed
  together rather than creating incomplete intermediate snapshots.
- All 60 retained code displays agree with rerun A1 preprocessing in timestamps,
  six used signals and quality labels. 54 protected A1 file hashes passed.
- A1 keeps >=20 usable sample-equivalent seconds per person/task. It removes both
  tasks for 22/39 and rectangle for 1/18. Cache retains 856,785 code rows from 31
  people; velocity >1,000 deg/s removes 138, leaving **856,647** (510,927 vehicle,
  345,720 rectangle). This is 3,426.588 nominal sample-seconds, not continuous time.
- There are 572 retained timestamp intervals >150 ms, maximum 25.678280 s. Never
  compress deleted holes into a continuous 250 Hz series.
- Historical results.md has obsolete 34/29-feature and eight-component PCA
  counts. Current A1 code/report use 32/27 and seven components. Do not rewrite
  that old log; PDF/HTML exports also predate the latest report.md changes.

### Cross-machine reproduction (Eivind, 2026-11-02)

- macOS arm64, Python 3.14.7, same pinned packages except Linux-only
  `nvidia-nccl-cu13` and `pip`. Rebuilt `dataset.parquet` with `cleaning.py`, then
  ran data.py, checks.py, quality.py (with and without `--apply`) and compare.py.
- All committed CSV evidence is byte-identical: window audit, displays, recordings,
  time parts, class quality. V2 predictions, participant metrics, search scores and
  inner folds are byte-identical; all summary metrics match exactly.
- Only parquet byte hashes differ (A1 cache `806de8…` vs `2c9a56…`; windows and V2
  tables likewise). Parquet bytes are platform-specific, so byte checks in
  checks.py and compare.py pass only on the machine that built the cache. Trym's
  provenance files were restored unchanged. V3 code verifies V2 by keys against
  committed CSV evidence and records portable content fingerprints instead.

## Window quality: V1 preserved, V2 chosen before new scores

Windows are non-overlapping, half-open two-second intervals anchored at raw screen
messages. They never cross people, segments or screen changes; partial final
windows are dropped. V1 requires >=90% support and <=150 ms continuously uncovered.

Time support assigns each retained sample at most 4 ms, clipped at the next sample
and window edge. `ok` is direct support, `interpolated` is filled support; the
remainder is uncovered, including small timing jitter. These are conservative
operational durations, not proof that every tracker-valid sample is correct.

Development V1 audit, before new model scores:

- 992 accepted windows: interpolation-time bins 0%, >0–5%, >5–10%, >10–20%, >20%
  contain **582, 152, 180, 47, 31** windows respectively.
- Interpolation p50/p90/p95/p99/max: **0/7.60/13.39/46.45/75.87%**.
- Total time: **1,911.754818 s direct, 66.651558 s interpolated, 5.593624 s uncovered**.
- Continuity alone admits interpolation-dominated summaries. V2 adds **>=80%
  direct support**, or 1.6 s in each window. This pragmatic threshold is not a
  validated physiological standard. Policy and expected loss were recorded in
  quality_policy.json and committed before the model comparison.
- Alternatives: retain V1 with quality stratification; require 90% direct support
  (more selection); discard all interpolation (strong selection against blinking).
  V2 keeps modest interpolation while excluding dominated windows. No score-based
  exclusions, threshold search, restored pid 1 or changed research question.

| Version | Development people/windows | Development vehicle/rectangle | Locked people/windows | Locked vehicle/rectangle |
|---|---|---|---|---|
| V1 | 23 / 992 | 562 / 430 | 7 / 374 | 228 / 146 |
| V2 | 23 / 961 | 546 / 415 | 7 / 369 | 224 / 145 |

V2 removes 31 development and five test windows; no additional person or display
is lost. V2 totals 1,330 windows from 30 people. Pid 1 already had zero V1 windows.
Test IDs stay **2, 6, 8, 18, 25, 28, 37**. Pid 18 has only vehicle; future test
reporting must not present that person's score as two-class discrimination.
These are test quality counts only, never test model results.

## Features and preprocessing decisions

A has 13 features, B has 11. A adds only mean screen x/y to B. Both contain x/y
population SD, left/right pupil mean and SD, speed median/p90, mean cached A1
blink rate and gap fraction, and interpolated-row fraction. Full allowlists are
in src/data.py. ID, label, stimulus, absolute time, segment, file, language, raw
eye coordinates, eye position and gaze vectors are never supplied as features.
B retains relative spatial structure; it is not spatially information-free.

Means/SDs are row-weighted, SD uses ddof=0. Speed uses within-window consecutive
pairs, actual dt, A1 pixels-per-degree geometry, dt <=20 ms and <=1,000 deg/s.
Cached blink/gap signals are A1's trailing **1,250 sample rows**, nominally five
seconds; their window means are not new blink counts or exact clock windows.

Pupils are not clipped using whole-person statistics. Inspected code ranges are
left 1.87–5.33 mm, right 1.70–4.76 mm; extremes are not proven artefact-free. An
alternative is fold-training-only common clipping bounds. Trees need no scaling;
kNN/SVM use StandardScaler inside each training pipeline. No PCA: the selected
13/11 summaries are small and interpretable, and PCA would mix the position
comparison. Original A1 behaviour is unchanged.

### V3 event features (Eivind, 2026-11-02)

feature_plan_v3.json fixed all thresholds before any V3 distribution or score.
src/fixations.py runs I-VT on the same V2 windows and samples: centred three-sample
speed, 30 deg/s, fixations >=60 ms, saccades >=2 samples, intervals <=20 ms. An
interpolated sample or longer interval ends any event, because A1 fills blinks
with straight lines. Events stay inside their window.

Eight features: fixation rate, mean duration, time fraction; saccade rate, mean
amplitude; horizontal, leftward (regressions plus line returns) and upward shares.
Empty-event means/shares are 0; rates keep that information. All use coordinate
differences only. Every real window was recomputed after a (128, -64) px screen
shift with identical events and features. A3 = A + 8 (21), B3 = B + 8 (19).

Development medians look physiological: 3.5 fixations/s, mean fixation 266 ms,
91.6% of time in fixation, 3 saccades/s, amplitude 1.37 deg. Four development and
19 locked-test windows have no qualifying fixation; these are counts only. Evidence
is in results/features_v3/; the feature table is ignored cache/windows_v3.parquet.

## Fixed model plan and evaluation

comparison_plan.json, committed in `2c2edf8` before new scores, specifies:

- Primary: mean two-class balanced accuracy, **equal weight per person**.
  Secondary: equal-person macro-F1, pooled window scores and class confusion.
- Reuse the original three outer person folds for all eight variants. V2 outer
  validation sets have 315, 342 and 304 windows (8, 8 and 7 people).
- Three inner person folds inside each outer training set select parameters.
  Never select on outer validation. First grid setting wins exact ties.
- Six settings per family/feature group: tree depth 2/4/6 x leaf size 5/10;
  kNN k=5/15/31 x uniform/distance weights; RBF-SVM C=.1/1/10 x gamma=.01/.1.
  This is 108 candidate searches / 324 inner fits, without adaptive expansion.
- Depth/leaf size limit tree complexity; k and weights compare local/smoother
  neighbours; C/gamma compare regularisation/kernel locality after scaling.
  Same budget is fairer than a tuned model against an arbitrary default, but
  does not optimise all families equally thoroughly.
- Tree/SVM class weights are balanced; kNN uses Euclidean distance. Training is
  window-weighted; equal-person evaluation does not equalise training influence.
- Untuned controls: majority dummy and position-only depth-one tree (min leaf 10).
  Controls are not equally searched competitors. A/B may choose different
  hyperparameters, so their difference is a pipeline comparison, not causation.
- Per-person CSV includes both class counts and TN/FP/FN/TP. Missing-class people
  retain predictions but have null two-class metrics and separate observed-class
  recall. All 23 current development people have both classes.
- No full-development final fit or final-test evaluation yet. The selected outer
  refits estimate the bounded selection procedure, not a final deployed model.

## V3 model and deep-model plans (declared before scores)

Committed before any V3, ensemble, sequence or CNN score:

- model_plan_v3.json: tree/kNN/RBF-SVM on A3/B3 with the V2 grids, jointly tuned
  with a mutual-information SelectKBest filter (k = 5, 10, all) fitted inside each
  training pipeline. AdaBoost (SAMME, 200 shallow balanced trees; depth 1/2/3 x
  learning rate .1/1) and bagged kNN (50 members, scaler+kNN; k 15/31 x feature
  share .5/.75/1). Boosting uses trees and bagging uses kNN, as point 7 asks.
  Same outer folds, inner partitions, selection rule, controls and timing as V2.
- deep_plan.json: 500-step, 6-channel sequences of the same V2 windows. A keeps
  absolute x/y; B centres x/y per window. Small 1D CNN, filters 16/32 x kernel
  5/9/15, early stopping on inner people, refit at median best epoch, seed 42 plus
  two spread seeds. Transfer: pretrain on instruction vs answer screens of the
  partition's training people only, then freeze or fine-tune (inner-selected).
- Locked test stays untouched until final_test_plan.json is committed.

## Executed commands and evidence

Run from repo root with existing .venv; no new packages were installed.

```bash
.venv/bin/python -B assignment3/src/checks.py
.venv/bin/python -B assignment3/src/quality.py
.venv/bin/python -B assignment3/src/quality.py --apply
.venv/bin/python -B assignment3/src/compare.py
```

- V1: results/data/ and results/first_validation/ remain unchanged. The old data
  builder and experiment.py reproduce V1; do not overwrite a historical run when
  changing analysis choices. Use a new version/directory for future revisions.
- Quality: results/quality/ has per-window time parts, full distributions,
  per-person/class selection counts and policy/hash evidence. Windows and derived
  feature tables stay ignored in cache/; no weights or raw samples are committed.
- Comparison: results/model_comparison_v2/ contains summaries, fold/person metrics,
  OOF predictions, all search scores, nested fold IDs, raw timing batches, plan,
  environment and checks. Environment records all package versions and hashes.

## Actual V2 results: development only

BA and F1 below average people, not folds or windows. Fold ranges are descriptive,
not confidence intervals; all variants use the same retained windows/folds.

| Model | Person BA | Person macro-F1 | Outer-fold BA range |
|---|---|---|---|
| Dummy | .5000 | .3520 | .500–.500 |
| Tree A | .6164 | .5837 | .573–.645 |
| Tree B | .5268 | .4704 | .507–.556 |
| kNN A | .5840 | .5484 | .575–.591 |
| kNN B | .4979 | .4688 | .459–.559 |
| RBF-SVM A | .5819 | .5266 | .567–.601 |
| RBF-SVM B | .5337 | .4959 | .458–.587 |
| Position-only stump | .5814 | .4534 | .565–.605 |

Tree A is highest in this bounded comparison, not proven universally superior.
Removing mean position reduces each family's mean here, but tuning choices also
change. B results are weak: kNN is below dummy on average, and SVM B has a .458
fold. Do not hide these negatives or expand searches to chase a desired result.

For the main question: with mean position, the best model reaches .616 and the
position-only stump .581; without it, the three families reach .498–.534 against
the .500 dummy. Here most usable signal is where people look, not how.

The stump uses mean y in all folds, with training thresholds about 233–252 px.
Its BA is near kNN/SVM A, but its errors differ: pooled rectangle recall .959 and
vehicle recall .167. Tree A's pooled recalls are .492 and .725. A similar BA does
not mean equally useful class behaviour. Pooled counts are secondary/window-
weighted; see participant_metrics.csv for actual per-person class support.
Tree A person BA ranges .414–.857. Three folds and 23 people give limited certainty.
These development people also appeared in V1 exploration. Nested selection limits
search optimism but does not turn them into fresh, untouched test people.
No result demonstrates comprehension, arbitrary-code generalisation, causation
or privacy protection after removing position. V1/V2 also differ in quality,
nesting and tuning; do not attribute score differences to quality alone.

## V3 basic models with feature selection: development only

`models.py basic` (2 min on a laptop) follows model_plan_v3.json; results are in
results/models_v3/basic/. Same people, windows, outer folds and inner partitions as
V2, so differences are paired by person. Verification PASS (sklearn recomputation
of 184 person rows, inner-only parameter choice, no locked-test predictions).

| Model | V3 person BA | person SD | Outer-fold range | V2 person BA |
|---|---|---|---|---|
| Tree A3 / B3 | .607 / .509 | .088 / .102 | .573–.657 / .489–.530 | .616 / .527 |
| kNN A3 / B3 | .535 / .528 | .138 / .108 | .491–.614 / .476–.615 | .584 / .498 |
| RBF-SVM A3 / B3 | .607 / .558 | .133 / .096 | .526–.690 / .504–.595 | .582 / .534 |
| Dummy / position stump | .500 / .581 | | | .500 / .581 |

- Event features plus selection do not give a consistent gain: SVM improves on
  both groups, tree falls slightly and kNN A falls. SVM B3 (.558) is now the best
  position-free model, but its fold range starts at .504. The A–B gap remains for
  tree and SVM; no B3 model is clearly above the position-only stump.
- The filter chose k = all in 11 of 18 family/group/fold cases; k = 5 or 10 picked
  different subsets per fold. Mutual information found no small stable subset.
- Mean pupil size is often in the MI top five. A plausible cause is screen
  luminance or task order (20 rectangle-first vs 13 vehicle-first), not reading;
  untested, so mention it only as a limitation.
- Permutation importance (descriptive): A3 relies on mean x/y and horizontal
  spread; B3 SVM on horizontal spread, saccade and fixation rates. B keeps relative
  spatial structure, e.g. line length, so B is not purely "how" people look.

## V3 boosting and bagging: development only

`models.py ensembles` (11 min, mostly permutation importance for bagged kNN);
results in results/models_v3/ensembles/. Verification PASS as above.

Plan amendment, before any ensemble score: the declared base-tree
`class_weight=balanced` crashed AdaBoost on its first inner fit ("worse than
random"). The tree optimises balanced weights, but AdaBoost scores it under
uniform weights, so a balanced stump can exceed 0.5 weighted error. Balance moved
to AdaBoost's initial window weights; grid and rounds unchanged. Recorded under
`amendments` in model_plan_v3.json.

| Model | Person BA A3 / B3 | person SD | Outer-fold range A3 / B3 |
|---|---|---|---|
| AdaBoost (trees) | .659 / .561 | .089 / .081 | .648–.672 / .527–.588 |
| Bagged kNN | .558 / .518 | .111 / .096 | .486–.612 / .487–.553 |

- AdaBoost A3 is the highest development score so far and the most stable over
  folds; it is .051 above the best basic A3 model and .078 above the position
  stump. A3 selected depth 2–3 with learning rate 1 in every fold; B3 picked the
  slow rate .1 in two of three folds.
- Bagged kNN improves on single kNN for A3 (.535 to .558) but not for B3 (.528 to
  .518); kNN remains the weakest family. Feature subsampling was chosen in four of
  six folds.
- B3 stays near .52–.56 for every family. Boosting raises A much more than B, so
  the A–B gap widens: ensembles extract more from position, not from dynamics.
- Cost (median refit / prediction of one outer validation fold): AdaBoost
  .77–.91 s / ~30 ms with 600–2,868 nodes; bagged kNN .13–.15 s / .27–.47 s,
  70–140x single kNN (3–4 ms), because all 50 members search their neighbours.
  V3 basic refits take ~.07–.10 s, mostly the mutual-information filter (V2 tree
  fit was 8 ms).

## Sequence CNN and transfer: development only

src/sequences.py builds 500-step, 6-channel sequences (x, y, two pupils, measured
mask, interpolated flag) for the same V2 windows; empty steps stay zero. 180 of
~661k code samples share a grid step and are dropped (counted). Transfer source:
658 instruction/answer-screen windows from the 23 development people (139
instruction, 519 answer screen; 753 rejected by the V1/V2 rules).

Plan amendment, before any sequence score: the declared +2 ms rounding put a
window's last sample on step 500; floor indexing is used instead (deep_plan.json
`amendments`). Implementation note: torch on macOS arm64 has no oneDNN, so
`nn.Conv1d` ran ~25x slower; cnn.py subclasses it with an unfold + einsum forward
(same parameters and initialisation; outputs agree to 3e-6).

`cnn.py` ran 3 h 50 min (results/deep_v3/). Verification PASS as for V3.

| Model | Person BA A / B | person SD | Outer-fold range A / B | seed SD (mean) |
|---|---|---|---|---|
| CNN from scratch | .687 / .483 | .110 / .152 | .620–.757 / .404–.550 | .022 / .022 |
| CNN + transfer | .684 / .499 | .105 / .094 | .635–.747 / .474–.521 | .020 / .017 |

- CNN A is the best development model (AdaBoost A3 .659); with absolute x/y the
  network learns where on screen the reading happens. CNN B is at or below chance
  in every fold: centred sequences carry no usable task signal for this network.
- Selected settings vary by fold (filters 16/32, kernels 5/9/15); refits used
  5–40 epochs. Seeds 43–44 move fold scores by ~.02, less than fold differences.
- Transfer does not help: .684 vs .687 with position, .499 vs .483 without.
  Fine-tuning won in five of six cases; one B fold stopped after two epochs. The
  source task (instruction vs answer screen) has only 255–322 windows per inner
  split and 430–447 per outer training set.
- Timing caveat: from about 14:20 three unrelated Python jobs used ~390% CPU and
  Low Power Mode was on, so CNN fit times after fold 1 are inflated (43% average
  CPU for the run). Item 10 uses the locked-test refit timings instead, measured
  on an otherwise idle machine.

## Locked-test plan (declared before any test prediction)

final_test_plan.json and src/final_test.py were committed after all development
results and before the test was touched. Every method (V2 basic, V3 basic with
selection, AdaBoost, bagged kNN, CNN, CNN + transfer, both controls; A and B) is
tuned with its own grid on three person folds over all 23 development people,
refit on all 961 development windows and scored once on 369 windows from the 7
locked people. Pid 18 (vehicle only) is excluded from two-class metrics, leaving
six people; 95% person-bootstrap intervals and paired A-minus-B differences are
descriptive. The script refuses to run if results/final_test/ exists.

## Locked-test results (run once, 2026-11-03)

`final_test.py` ran once in 4 min 55 s on an otherwise idle laptop (battery, Low
Power Mode off); checks PASS. Six two-class people; pid 18 vehicle recall reported
separately. Intervals are 95% person-bootstrap, descriptive only.

| Method | Test BA A / B | 95% interval A / B | Paired A − B (interval) | A > B |
|---|---|---|---|---|
| Tree V3 | .728 / .584 | .664–.791 / .532–.654 | .144 (.029–.254) | 5/6 |
| kNN V3 | .622 / .523 | .581–.657 / .475–.559 | .099 (.041–.147) | 5/6 |
| RBF-SVM V3 | .632 / .582 | .590–.672 / .542–.632 | .050 (−.020–.122) | 4/6 |
| AdaBoost | .713 / .575 | .677–.758 / .531–.623 | .138 (.077–.204) | 6/6 |
| Bagged kNN | .552 / .504 | .484–.611 / .470–.538 | .048 (−.010–.101) | 4/6 |
| CNN | .753 / .527 | .709–.796 / .483–.590 | .227 (.157–.289) | 6/6 |
| CNN + transfer | .759 / .458 | .724–.796 / .401–.531 | .301 (.236–.379) | 6/6 |
| Majority / position stump | .500 / .582 | | | |

V2 basic models on test: tree .726/.510, kNN .565/.520, SVM .625/.531.

- The main finding holds on fresh people: with position, CNN, tree and AdaBoost
  reach .71–.76; without it, the best is .584 (tree B3), level with the
  position-only stump (.582). Every family is better with position; for five of
  seven the paired interval excludes zero.
- Test scores with position are higher than development (CNN .753 vs .687, tree
  .728 vs .607). Six people cannot show whether this is luck or easier recordings;
  development remains the main comparison and no model was re-chosen.
- Refit on 961 windows: V3 trees and kNN .02–.03 s (mostly the MI filter), SVM .04 s, AdaBoost .3–.5 s,
  CNN 0.7–5.9 s; predicting 369 test windows: tree <1 ms, kNN/SVM 1–9 ms,
  AdaBoost 8 ms, bagged kNN 66–164 ms, CNN 35–44 ms on one thread.
- CNN seed spread on test: .720–.756 (A), .527–.556 (B), transfer similar.
- Without contention the full test, including CNN tuning on three development
  partitions, took under five minutes, versus 3 h 50 min for the development CNN
  run: the development CNN timings are dominated by the CPU contention noted above.

## Time, complexity and verification

Timing separates inner-search wall time from repeated selected-outer-refit fit
and prediction. Warm up, then five batches each with >=20 ms timed work or the
1,000-call cap. Fresh-model cloning, I/O and metrics are excluded; scaler work is
included. Raw batch counts/times and per-fold quartiles are saved. These are warm
validation-batch costs, not end-to-end real-time deployment latency.

| Model A | Total inner search, 3 outer folds | Median outer fit | Median prediction batch |
|---|---|---|---|
| Tree | .808 s | 7.94 ms | .164 ms |
| kNN | .823 s | 4.23 ms | 10.50 ms |
| RBF-SVM | 1.327 s | 24.92 ms | 9.23 ms |

Prediction batches contain 304–342 windows; both groups/controls have full timing
rows. Tree A depth is 4–6 with 17–37 nodes; kNN retains 619–657 training windows;
SVM A uses 409–627 support vectors. More expensive prediction is consistent with
neighbour/kernel evaluation, but this small warm benchmark is not a scaling law.

Checks PASS: A1/V1 hashes, time-part conservation and synthetic holes, V2 eligibility,
nested person separation, training-only scaler means/sample counts, input
allowlists, one-class metric suppression and unique OOF rows. Independent sklearn
recomputation verifies all 184 participant metric/confusion rows, 7,688 predictions,
24 outer results and the 108 inner-only parameter selections. All 15 old result
files are unchanged; no locked test predictions exist.

The first timing audit incorrectly required every batch to reach 20 ms. Six
batches hit the declared 1,000-call cap sooner. The audit was corrected to the
predeclared budget-or-cap rule; raw results were retained, no models rerun.
See results/model_comparison_v2/verification.json. Source and plan hashes match.

## Open questions and next work

1. Ask course staff what **transfer** in point 8 requires and whether point 7
   requires different base-learner families. Neither is assumed resolved/complete.
   Also clarify whether the deep model can count among point 5's three algorithms.
2. Review these errors and limitations with Eivind. Then predeclare a bounded
   boosting/bagging comparison with appropriate distinct base learners, on both
   feature groups. Do not infer a best learner from tiny validation differences.
3. Later implement/tune a small DNN or CNN, with grouped evaluation and justified
   hyperparameters. Mirror A/B: absolute x/y versus x/y centred per window.
   Do not start this larger part in the current session.
4. Freeze analysis/model choices, evaluate the same locked test once, compare all
   required methods and write the final report.

Eivind may reconsider operational quality thresholds, future feature/ensemble
choices and search budgets with reasons recorded before new scores. A future
quality sensitivity study must preserve old versions and use development data
only. More participants/tasks and exact clock-based blink features are optional
improvements, not excuses to change the agreed question. Pid 1 remains excluded
by quality, Scala has only one participant, and quality selection/correlated
windows restrict the population to which conclusions apply.
