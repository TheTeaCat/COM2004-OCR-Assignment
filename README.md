# COM2004-OCR-Assignment

This was an assignment to perform OCR given several pages of text and their 
bounding boxes, one set of which was entirely clean, and the other of which was
progressively noisier.

 - A description of the assignment can be found in [Provided_README.md](/Provided_README.md)

 - A description of my solution can be found in [report.md](/report.md)

**All files except this readme form the assignment.**

Those modified/created by me were:
 - [code/system.py](/code/system.py)
 - [code/data/model.json.gz](/code/data/model.json.gz)
 - [report.md](/report.md)
 - [Results.xlsx](/Results.xlsx) (although this was not part of the submission)

## Setup

See [Provided_README.md](/Provided_README.md)

## Performance

Performance on the provided test pages can be found in [report.md](/report.md).

Performance on the unseen test pages (not provided) used to mark the assignment 
were as follows:

>### Scores on test pages:
>
>| Page   | % Correct |
>|--------|-----------|
>| Page 1 |    96.3   |
>| Page 2 |    96.6   |
>| Page 3 |    94.1   |
>| Page 4 |    82.9   |
>| Page 5 |    72.1   |
>| Page 6 |    59.5   |
>
>**Average correct = 83.6%**