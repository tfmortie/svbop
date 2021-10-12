# Efficient Set-valued Prediction in Multi-class Classification

*Python and C++ implementation for [DAMI paper](https://doi.org/10.1007/s10618-021-00751-x) about efficient set-valued prediction in multi-class classification.*

## Getting started

Implementation is provided as a PyTorch (Python) extension and in C++.

### PyTorch (Python)

* Soon to come

### C++

Implementation of SVBOP-CH (flat classification) and RSVBOP-CH (hierarchical classification) can be found in **src/main/cpp/models/hierarchical.cpp**. 

**Important**: make sure to build with flags `-O3 -mtune=native -march=native`, in order to have optimal runtime performance. This is necessary for the Eigen library, which is used for (efficient) mathematical operations.

## Running code 

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
        -o, --optim             Algorithm to use in the optimization problem
                0 := SGD (stochastic gradient descent)
                1 := Adam (adaptive moment estimation)
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
        -f, --file              File path and prefix for saving predictions (if specified)  
        -s, --seed              Seed for random engines       
```
