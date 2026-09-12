# Data preprocessing

> Made by Eivind Systad Geiran
> 
> Started on: Tuesday September 8th

## Data preprocessing

### Data exploration
> This has been done mostly on "1_rawdata.tsv", with some random checks to other files
>
> I have also done a more overview type of analysis to know how big each file is and general content

- Around 52000 measurements per file (on average)
- About 11.5 messages per file

**Measurements**
- A lot of columns have only 1 value
  - Aux1 (has 0 values for some of the datasets (such as 1_radata.csv))
  - Trial
  - R Plane
  - L Plane
  - Timing
  - Frame
- Some have very few unique values, these might be really important 
  - R Validiy (2)
  - L Validity (2)
  - Pupil Confidence (3)
- This dataset only has numbers
  - All of them are of type float64 except for Time and Trial. 
  - L Raw X is an object if you do not filter the dataset before finding types
    - This is because messages are strings and it is a float for measurements
  - Some of them can be classified as categorial due to few unique values
    - Validity (true/false)
    - Pupil Confidence (0, 1, 2)

**Messages**
- Most messages are pictures
  - instruction_calibration.jpg
  - instruction_comprehension.jpg
  - ...
- Some are mouseclicks 
  - They also give you x and y value of the clicks
- These are all categorical in my opinion
  - The mouseclicks carry some data, so they might not be, but in that case you should move the X and Y away from the string and the message will be categorical

- Holes:
  - I found varying amount of good data - some sets had a lot of bad data. 
  - I found 9 sets with over 10% of the recording being wasted at not looking at the screen.
    - This was found using "L Validity"   
  - I will explore what to do with these in the next part


### Data cleaning
> This will be based on the findings in the exploration
> Due to the holes in file 1 I will primarily use 2 for testing during cleaning. 

> *Finding:* In this dataset you should probably not use mean or median data to cover for NaN or wrong data as that would probably not be where the person actually looks (or remotely close for that matter) - maybe we can consider it for some columns if they are missing values. 

- Missing values in the dataset:
  - Aux1 - almost no data -> remove the column in all datasets

**What counts as a valid sample**

The exploration numbers above were based on `L Validity` alone. The final definition marks a
sample invalid if *any* of these hold, and the last two turned out to matter:

1. `L Validity == 0` or `R Validity == 0`
2. either eye's POR outside the 1920x1080 screen
3. either eye's POR at exactly 0 (the tracker's "no data" sentinel)

- (2) and (3) catch samples the tracker itself calls valid. Condition (3) alone accounts for
  8223 samples spread over 31 of the 32 tsv files (0.50% of all samples).
- The right eye is lost noticeably more often than the left, so the `L Validity`-only figures
  understated the loss - file 9 went from 19% to 41%, file 41 from 17% to 32%, file 34 from
  5% to 16%.

**Handling the holes**

Holes are found per stimulus segment (the MSG rows), never across a change of stimulus.
A hole shorter than 150 ms is linearly interpolated against `Time`; anything longer is kept
as NaN and labelled a gap. Every sample gets a `quality` column of `ok` / `interpolated` /
`gap`, so it stays visible later how much of the analysis rests on invented values.

Rationale for the 150 ms cut-off: gaze does not move during a blink, so the endpoints of a
short hole are a good estimate of what happened inside it. Over a multi-second loss the
participant may have looked anywhere, and a straight line between the endpoints would be
fiction.

150 ms rather than 100 ms so that the threshold agrees with the `blink` category: at 100 ms
holes were being *classified* as blinks and then *treated* as unrecoverable loss, which is
incoherent. Averaged over the 33 participants the change moves 1.42 percentage points from
gap to interpolated (mean gap 16.19% -> 14.77%), and the result is not sensitive to the exact
value - anywhere from 150 to 250 ms gives the same output.

No margin is trimmed around the blinks. Partial eyelid occlusion on the bordering samples
would have argued for one, but the pupil diameter column is clean on every row that survives
cleaning (1.87-5.68 mm, no zeros), and post-blink diameter sits within 0.5% of the pre-blink
baseline, so there is nothing to trim.

Loss per participant with the final definition (33 files, 211 included):

| file | samples | ok % | interpolated % | gap % |
|---|---|---|---|---|
| 22 | 51401 | 1.8 | 0.9 | **97.3** |
| 39 | 48064 | 31.1 | 2.4 | **66.5** |
| 18 | 61335 | 43.4 | 1.6 | **55.0** |
| 1 | 53233 | 45.1 | 8.2 | **46.7** |
| 23 | 40453 | 57.3 | 10.0 | **32.7** |
| 41 | 81685 | 67.7 | 2.5 | **29.8** |
| 9 | 48294 | 58.9 | 14.4 | **26.7** |
| 34 | 56080 | 84.4 | 0.5 | 15.2 |
| 20 | 48746 | 85.3 | 0.7 | 13.9 |
| 24 | 46509 | 85.7 | 1.4 | 12.9 |
| 8 | 63784 | 84.8 | 3.9 | 11.3 |
| 11 | 40313 | 88.4 | 0.7 | 11.0 |
| 37 | 57051 | 89.2 | 0.7 | 10.1 |
| *(20 remaining files)* | | | | < 10 |

**Excluding data: per (participant, stimulus), not per participant**

`gap %` measures the whole session, including breaks, calibration and instruction screens.
What we actually model is code reading, so the two metrics disagree badly:

| pid | vehicle s | rectangle s | usable code s | gap % |
|---|---|---|---|---|
| 22 | 1.1 | 2.2 | 3.3 | 97.3 |
| 39 | 15.4 | 6.5 | 21.9 | 66.5 |
| 1 | 45.4 | 12.6 | 58.0 | 46.7 |
| 3 | 36.5 | 28.5 | 65.0 | 7.9 |
| 18 | 92.3 | 5.0 | 97.3 | 55.0 |
| 41 | 65.6 | 40.9 | 106.5 | 29.8 |
| 7 | 106.9 | 87.7 | 194.6 | 5.8 |

- Participant 41 loses 30% of the session and still has 106.5 s of code reading, above the
  median. A percentage cut-off would have removed one of the better recordings.
- Participant 1 loses 47% but its largest single gap is only 6.0 s (6% of its loss), so the
  loss is spread-out blinking rather than a break.
- Participant 18 is not noisy in general - it has 92.3 s on vehicle and 5.0 s on rectangle.
  That is a missing stimulus, not a bad participant.

Cells are therefore kept if they hold at least 20 s of usable data (`coverage()`), which drops
**6 of 66 cells**: 22 and 39 lose both stimuli and so disappear entirely, while 1 and 18 lose
only their rectangle cell. Excluding on `gap %` at 25% would have dropped 7 whole participants.

**Hole taxonomy**

The original two-way blink/loss split at 150 ms did not survive contact with the
distribution (note this 150 ms is unrelated to `SHORT_MS`, which happens to share the value): 79% of all
holes are under 20 ms and the median hole is 8 ms, i.e. two samples. Those are sensor dropouts,
not blinks - counting them gave a blink rate of 26/s, which is physiologically impossible.

| bucket | share of holes | share of lost samples |
|---|---|---|
| < 20 ms | 78.8% | 10.7% |
| 20-50 ms | 7.7% | 4.9% |
| 50-100 ms | 3.5% | 5.0% |
| 100-150 ms | 2.6% | 6.5% |
| 150-400 ms | 5.2% | 23.4% |
| > 400 ms | 2.1% | 49.5% |

Holes are now split three ways: `dropout` (< 50 ms, sensor noise), `blink` (50-400 ms, the
physiological range) and `loss` (> 400 ms, track loss or looking away). For file 2 that gives
18 blinks with a median of 122 ms - a textbook blink duration - against 347 dropouts at 4 ms.
Note the last two rows: 7% of the holes account for 73% of all lost samples.

**Using the gaps instead of only removing them**

Where the eye was during a gap is unrecoverable, but *that* there was a gap is behaviour. Two
features are extracted before the gap rows are dropped: `blink_rate` and `gap_fraction`, both
over a trailing 5 s window. Blink rate is an established index of visual attention demand, and
long losses usually mean the participant looked away rather than that the sensor failed.

They are computed over a trailing window rather than per stimulus on purpose: a per-stimulus
mean would encode the target and leak into the model. Blink rate now ranges 0-0.4 blinks/s.

**The resulting dataset**

`build_dataset()` assembles the cleaned per-participant frames into one table, applies the
coverage rule, drops the `gap` rows and caches the result as parquet (the full parse takes
~30 s, and every later step re-reads the cache instead).

| | |
|---|---|
| rows | 1 440 469 |
| columns | 46 |
| participants | 33 (31 with code-reading data) |
| code-reading rows | 856 785 (511 009 vehicle / 345 776 rectangle) |
| interpolated | 3.7% of surviving rows |
| size | ~100 MB parquet, ~490 MB in memory |

Participants 22 and 39 contribute no code-reading data, as intended. For a thin cell only the
code-reading screen is dropped, not the matching multiple-choice screen, since the exclusion
is about the quality of the reading data.

**Stimulus labels**

Each sample is labelled with the stimulus on screen, derived from the MSG rows. Only `.jpg`
messages change the screen - a `UE-mouseclick` message is an event on the screen that is
already showing, so treating it as a boundary fragmented the multiple-choice segments (in
file 2 it split 464 samples off `mupliple_choice_rectangle`). Clicks are ignored for
segmentation, which leaves 8 clean segments per participant.

**211_rawdata.xlsx**

The 33rd participant is stored as Excel rather than tsv, but the content is identical - same
header block, same 45 columns, same stimulus sequence, 50739 samples. It is read via
`load_xlsx()` instead of being dropped: at 3.3% gaps it is one of the cleaner recordings, and
excluding it for a file-format reason would not be defensible.
-

## Handling outliers

Two detectors, on the two columns where the methods are actually valid, plus an explicit
argument for not using IQR on the gaze coordinates. Code in `outliers.py`.

**Saccade velocity (Z-score is not needed - the limit is physical)**

The screen geometry is identical in all 33 headers: 1920x1080 px on 344x194 mm at 700 mm
viewing distance. That makes the threshold derivable rather than guessed:

```
px/mm       = 1920 / 344       = 5.58
1 deg at 700 mm = 2*700*tan(0.5 deg) = 12.22 mm  ->  68.2 px/deg
1000 deg/s  = 68 200 px/s      = 273 px per 4 ms sample
```

No human saccade exceeds ~1000 deg/s, so anything above it is a measurement artefact rather
than a fast eye. Velocity is computed as `hypot(dPOR) / dt`, grouped by `(pid, segment)` so no
difference crosses a participant or a stimulus change, and `dt` comes from `Time` so the gaps
dropped earlier cannot masquerade as instant jumps.

Result: **178 of 1 440 469 samples (0.012%)** exceed the limit; median speed is 6.6 deg/s, p99
is 244 deg/s and the maximum is 2061 deg/s. The p99 is the reassuring number - it sits in the
normal saccade range, so the detector is catching the tail and not the signal.

*Decision: remove.* An impossible velocity means the sample's coordinates are wrong, and at
0.012% the cost of deleting them is nil. Velocity is kept as a feature (`velocity_deg_s`).

**Pupil diameter (Z-score, per participant)**

Baseline pupil size differs substantially between people, so a corpus-wide Z-score would flag
every participant with naturally large pupils instead of finding artefacts. It is therefore
computed within participant. At |z| > 3, **16 425 samples (1.14%)** are outliers - against
0.27% for a normal distribution, which confirms the heavy tail.

Both prescribed methods are applied to this column, since it is the one where both are valid:

| | global IQR | per-pid IQR | per-pid Z>3 |
|---|---|---|---|
| `L Mapped Diameter` | 3.09% | 1.84% | 0.68% |
| `R Mapped Diameter` | 0.88% | 1.95% | 0.85% |

Two things follow. First, the **global IQR disagrees between the eyes by a factor of 3.5**
(3.09% vs 0.88%) while the per-participant IQR agrees closely (1.84% vs 1.95%) - the global
fence is being distorted by mixing participants with different baseline pupil sizes, which is
the same argument as for the Z-score and now shown rather than asserted. Second, IQR is the
more sensitive of the two here: it flags roughly 2.5x as many samples, and only 14 of the
16 425 Z-outliers fall outside the IQR fence, so **the Z set is essentially nested inside the
IQR set**. That is expected - with a heavy tail the outliers inflate the standard deviation
and partly hide each other, whereas the quartiles barely move.

Z-score at |z| > 3 is used for the capping because it is the more conservative of the two, and
the capped value has a direct interpretation (3 standard deviations from that participant's own
mean).

*Decision: cap, do not remove.* Pupil dilation is a real physiological signal and one of the
more informative columns here; deleting its extremes would delete the effect a model is meant
to find. Clipping to +/-3 sd within participant keeps the row and bounds the leverage.

*Why not transform.* The third option in the task is transformation, normally a log to pull in
a long right tail. It is not warranted: skew is 0.82 and the full range is 1.87-5.68 mm, a
factor of 3. A log would compress a distribution that is already close to symmetric and would
cost the interpretability of a column measured in millimetres.

**Why not IQR on the gaze coordinates**

Reading happens in the left half of the screen, so the coordinate distribution is spatially
structured rather than unimodal-with-tails, and Tukey's fence lands *inside* the stimulus:

| column | IQR fence | screen |
|---|---|---|
| `L POR X [px]` | [409, 1389] | 0-1920 |
| `L POR Y [px]` | [-127, 968] | 0-1080 |

The X fence covers 51% of the screen width, which makes the entire right 28% of the screen
"outlier" territory. It flags 14 399 samples, and every one of them is inside the screen and
passed the validity check - they are real gaze points. Worse, the largest group of them
(6253 samples) falls on `instruction_calibration.jpg`, whose targets are deliberately placed
at Position(96;810), Position(1824;270) and the other corners. IQR would reject the
calibration data for being exactly where the calibration asked the participant to look.

The Y fence shows the same problem from the other side: its lower bound is negative, so it
cannot flag anything below the midline at all.

The distinction is the point. The velocity detector flags transitions that are *physically
impossible*; IQR on coordinates flags positions that are merely *off-centre*. Only the first
is an outlier.

