# Set-Valued Prediction in Flat and Hierarchical Multi-Class Classification

*Python and C++ implementation (in progress) for [arXiv paper](https://arxiv.org/abs/1906.08129) about set-valued prediction in multi-class classifcation.*

## Getting started

Implementation is provided in C++ and Python.

### C++

*Under construction.*

### Python

* HNet is implemented in **src/main/py/models/hclassifier.py**, which contains the code for the hierarchical probabilistic neural network model, as well as the RBOP
inference algorithm. 

* FNet is implemented in **src/main/py/models/fclassifier.py**, which contains the 
code for the flat probabilistic neural network model, as well as the UBOP 
inference algorithm.

## Running code 

### C++

*Under construction.*

### Python

#### Training HNet and FNet

You can train HNet and FNet as follows:

```
export CUDA_VISIBLE_DEVICES=0 # use GPU

python -u train.py -pcsv ./data/VOC2006/TRAINVAL.csv -m hierarchical -s $( cat ./data/VOC2006/hierarchy_full.txt ) --gpu --vgg --no-ft 

python -u train.py -pcsv ./data/VOC2006/TRAINVAL.csv -m flat --gpu --vgg --no-ft 
```

For HNet, you can also train on, e.g. k=10, random generated structure as follows:

```
export CUDA_VISIBLE_DEVICES=0 # use GPU

python -u train.py -pcsv ./data/VOC2006/TRAINVAL.csv -m hierarchical -k 10 --learns --gpu --vgg --no-ft 
```


#### Testing HNet and FNet

Subsequently, you can test HNet and FNet as follows:

```
export CUDA_VISIBLE_DEVICES=0 # use GPU

python -u test.py -m flat -pm ./FLAT_MCC_TRAIN_0.2_True_100_0.001_32_5_False_True_VOC2006_VOC2006.pt -pcsv ./data/VOC2006/TEST.csv --gpu --vgg --store 

python -u test.py -m hierarchical -pm ./HIERARCHICAL_MCC_TRAIN_0.2_True_100_0.001_32_5_False_True_VOC2006_19_False_0_VOC2006.pt -pcsv ./data/VOC2006/TEST.csv -s $( cat ./data/VOC2006/hierarchy_full.txt ) --gpu --vgg --store 
```

**Important**: make sure to first train HNet and FNet before testing, as trained models need to be provided. After training, trained models are automatically saved to the working directory. 
