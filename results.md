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
A hole shorter than 100 ms is linearly interpolated against `Time`; anything longer is kept
as NaN and labelled a gap. Every sample gets a `quality` column of `ok` / `interpolated` /
`gap`, so it stays visible later how much of the analysis rests on invented values.

Rationale for the 100 ms cut-off: a blink lasts roughly 100-150 ms, and gaze does not move
during one, so the endpoints of a short hole are a good estimate of what happened inside it.
Over a multi-second loss the participant may have looked anywhere, and a straight line between
the endpoints would be fiction.

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

Seven participants lose more than 25% of the recording. TODO: decide and justify a cut-off
for excluding participants (22 and 39 are clearly unusable).

**211_rawdata.xlsx**

The 33rd participant is stored as Excel rather than tsv, but the content is identical - same
header block, same 45 columns, same stimulus sequence, 50739 samples. It is read via
`load_xlsx()` instead of being dropped: at 3.3% gaps it is one of the cleaner recordings, and
excluding it for a file-format reason would not be defensible.
- 