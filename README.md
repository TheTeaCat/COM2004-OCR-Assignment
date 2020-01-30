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

If you're lazy (I am too):

```
python train.py
python evaluate.py dev
```

## Performance

Performance on the unseen test pages (not provided) and the provided test pages 
were as follows:

| Page   | % Correct (unseen) | % Correct (provided) |
|--------|--------------------|----------------------|
| Page 1 |        96.3        |         97.6         |
| Page 2 |        96.6        |         98.7         |
| Page 3 |        94.1        |         96.5         |
| Page 4 |        82.9        |         86.8         |
| Page 5 |        72.1        |         74.0         |
| Page 6 |        59.5        |         63.8         |
|  Avg.  |        83.6        |         86.2         |

## Other Solutions

I've found one other solution to this assignment on GitHub from my class:

 - [@xuan525/OCR-System-Assignment](https://github.com/xuan525/OCR-System-Assignment)

## Provided Pages

|                Page 1                |                Page 2                |                Page 3                |
|--------------------------------------|--------------------------------------|--------------------------------------|
| ![Page 1](/code/data/dev/page.1.png) | ![Page 2](/code/data/dev/page.2.png) | ![Page 3](/code/data/dev/page.3.png) |

|                Page 4                |                Page 5                |                Page 6                |
|--------------------------------------|--------------------------------------|--------------------------------------|
| ![Page 4](/code/data/dev/page.4.png) | ![Page 5](/code/data/dev/page.5.png) | ![Page 6](/code/data/dev/page.6.png) |
