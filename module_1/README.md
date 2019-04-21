
## Assignment 1

### MNIST Classifier via bare numpy

Metric: averaged f1-measure over all classes

kernel svm, test average f1: 0.96

Usage:

run `gzip -dk *` in dataset folder

`python train.py —x_train_dir=\</dir> —y_train_dir=\</dir> —model_output_dir=\</dir>`

`python predict.py —x_test_dir=\<dir> —y_test_dir=\</dir> —model_input_dir=\</dir>`
