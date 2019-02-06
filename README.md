
# MNIST WITH KNN
> 1. Using knn to classify 8400 testing pictures based on 33600 training pictures with Python in numpy. 
> 2. Using knn to classify 28000 testing picture based on 42000 training pictures and submit the result on kaggle.

> ### Training and testing files are available in http://yann.lecun.com/exdb/mnist/
> ### require packages:
>>> * numpy
>>> * time
>>> * csv

>### The code also calculates real time costs for each part and uses package pickle to store the sorted neighbors and distance matrix
 
>### The results include accuracy rate and confusion matrix




## Usage example

For experiment 1, run the mnist_csv_split_training.py file first. You can get a nearest neighbor matrix nearest_split.csv file and the result based on k = 20. Then you can run mnist_change_k.py in order to compare the difference among different k vaules in short time.

For experiment 2, run mnist_csv_submit.py file. Then submit the generated result.csv file directly to the kaggle website.

## Development setup

Download the files and run directly, and make sure put the source files train.csv and test.csv under the same director.




