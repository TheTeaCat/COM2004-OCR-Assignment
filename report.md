# OCR assignment report

## Feature Extraction (Max 200 Words)
I stuck with a standard PCA approach, with the following adjustments:

 - Centring the characters in their feature vectors instead of them being squished into a corner. This allows the covariance matrices to better distinguish between characters as there will be less overlap.
 - Not discarding the first covariance matrix as this data has no lighting, so it's still useful.
 - Subtracting the mean of the data I'm classifying instead of that of the training data. I believe this performs better as if a pixel was intended to be white across all feature vectors, they'd be better centred by subtracting their mean as opposed to the mean of the corresponding pixels in the training data. 
 - Adding noise to the training data in the same manner as the test data, with an x of 127 (highest x can get without creating any overlap between absolute black and absolute white), to make it more reflective of the test data.
 - Using a median blur filter for noise reduction as it better preserves edges. This is with an irregular footprint, that I've found works better than a standard square as I think it better preserves characters' rounded edges.

## Classifier (Max 200 Words)
I used a k-nearest neighbours classifier. 

I found on the last page, about ~200 nearest neighbours worked best, but with the first page 1 neighbour worked best. The third page worked best with about 5 neighbours. I decided to scale the nearest neighbours in exponential proportion to the variance of the margins of the page. This gave me about +4% overall performance at the time, with an improvement of 12.9% on page 6.

As a distance measure, I attempted using euclidean distance and cosine distance. Cosine distance performed significantly better on the noisier pages (improving page 5 by 18.5% over euclidean).

## Error Correction (Max 200 Words)
I initially made error corrections for apostrophes, commas, full stops, "l"s, "I", and "i"s based upon their bounding box positions in relation to the average midpoint of their surrounding characters on the same line. I then attempted to use quadgrams to make corrections, by combining the probabilities given by the quadgrams with the confidence of my k-nn classification, but this only made performance worse.

## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
- Page 1: 97.6%
- Page 2: 98.7%
- Page 3: 96.5%
- Page 4: 86.8%
- Page 5: 74.0%
- Page 6: 63.8%

These may differ slightly on retraining as the training data has random noise added to it.

## Other information (Optional, Max 100 words)

