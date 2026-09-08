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

> *Finding:* In this dataset you should probably not use mean or median data to cover for NaN or wrong data as that would probably not be where the person actually looks (or remotely close for that matter) - maybe we can consider it for some columns if they are missing values. 

- Missing values in the dataset:
  - Aux1 - almost no data -> remove the column in all datasets
- 1, 9, 18, 20, 22, 23, 24, 39, 41
  - 1: 45%
  - 9: 19%
  - 18: 53%
  - 20: 10.7%
  - 22: 91%
  - 23: 39%
  - 24: 13%
  - 39: 68%
  - 41: 17%
- 