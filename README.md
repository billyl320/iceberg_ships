# Classification of Synthetic Aperture Radar Images of Icebergs and Ships using Random Forests Outperforms Convolutional Neural Networks
Synthetic aperture radar (SAR) is a common technique for capturing vessels and icebergs on the ocean surface.  Convolutional Neural Networks (CNNs) are a popular approach to classify classes captured in images which include ships and icebergs.  However, CNNs are difficult to explain and are computationally expensive.  In this paper, we built a random forest (RF) model which outperforms CNN based approaches by about 7\% and 16\% on the testing and validation data, respectively.  The RF model used interpretable metrics.  These powerful metrics provide insight to what is important to distinguish the two classes from one another.  Thus, despite noise present in the data, the RF model was able to provide meaningful classifications between ships and icebergs captured by SAR.  

## Code
This branch provides the code for the model, some comparitive studies, and image preprocessing.  The image preprocessing was done in Python, while everything else was done in R.  

## Data
This branch provides the final images after preprocessing the images from https://www.kaggle.com/c/statoil-iceberg-classifier-challenge.  We also provide the txt files of our shape metrics.  
