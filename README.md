# Set-Valued Prediction in Multi-Class Classification

*Python and C++ implementation for [arXiv paper](https://arxiv.org/abs/1906.08129) about set-valued prediction in multi-class classification.*

## Getting started

Implementation is provided in C++ and Python.

### C++

* Implementation of the hierarchical model, together with UBOP and RBOP, can be found under **src/main/cpp/models/hierarchical.cpp**.
* Implementation of the flat model, together with UBOP and RBOP, can be found under **src/main/cpp/models/flat.cpp**.

**Important**: make sure to build with flags `-O3 -mtune=native -march=native`, in order to have optimal runtime performance. This is necessary for the Eigen library, which is used for (efficient) mathematical operations.

### Python

* Implementation of the hierarchical model (HNet), together with UBOP and RBOP, can be found under **src/main/py/models/hclassifier.py**.
* Implementation of the flat model (FNet), together with UBOP and RBOP, can be found under **src/main/py/models/fclassifier.py**.

## Running code 

### C++

Usage: 
```
svp <command> <args>

    command:
        train                   Training mode
        predict                 Predict mode
        -h, --help              Help documentation
    
    args:
        -i, --input             Training/test data in LIBSVM format
        -t, --type              Probabilistic model type
                0 := softmax
                1 := hierarchical softmax with slow updates
                2 := hierarchical softmax with fast updates
        -o, --optim             Optimizer
                0 := SGD
                1 := Adam
        -s, --struct            Structure classification problem
        -b, --bias              Bias for model 
              >=0 := bias included 
              <0  := bias not included 
        -ne, --nepochs          Number of epochs
        -lr, --learnrate        Learning rate 
        -bs,  --batchsize       Mini-batch size
        -ho, --holdout          Holdout percentage for fitting
        -pa, --patience         Patience for early stopping
        -d, --dim               Number of features of dataset (bias not included)
        -u, --utility           Utility function (format: {precision|recall|fb|credal|exp|log|reject|genreject})
        -p, --param             Parameters for utility (format: valparam1 valparam2 ...)
        -m, --model             Model path for predicting/saving
        -f, --file              File path for saving predictions (if specified)  
        -s, --seed              Seed for random engines       
```

### Python

#### Training HNet and FNet

Example code for training HNet and FNet:
```
python -u train.py -pcsv ./data/VOC2006/TRAINVAL.csv -m hierarchical -s $( cat ./data/VOC2006/hierarchy_full.txt ) --gpu --vgg --no-ft 
python -u train.py -pcsv ./data/VOC2006/TRAINVAL.csv -m flat --gpu --vgg --no-ft 
```
For HNet, you can also train on, e.g. k=10, random generated structure as follows:
```
python -u train.py -pcsv ./data/VOC2006/TRAINVAL.csv -m hierarchical -k 10 --learns --gpu --vgg --no-ft 
```

#### Testing HNet and FNet

Example code for testing HNet and FNet:
```
python -u test.py -m flat -pm ./FLAT_MCC_TRAIN_0.2_True_100_0.001_32_5_False_True_VOC2006_VOC2006.pt -pcsv ./data/VOC2006/TEST.csv --gpu --vgg --store 
python -u test.py -m hierarchical -pm ./HIERARCHICAL_MCC_TRAIN_0.2_True_100_0.001_32_5_False_True_VOC2006_19_False_0_VOC2006.pt -pcsv ./data/VOC2006/TEST.csv -s $( cat ./data/VOC2006/hierarchy_full.txt ) --gpu --vgg --store 
```
**Important**: after training, trained models are automatically saved to the working directory. 
