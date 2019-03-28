# Set-Valued Prediction in Flat and Hierarchical Multi-Class Classification

## Getting started

* Make sure to download VOC2006 images (example dataset) and place under **/data/PNGImages**.

* The random hierarchy generator code by Melnikov et al., can be found under **/models/nd/**.

* Current implementation of the hierarchical neural network HNet is not an exact hierarchical softmax implementation, but rather a tree-based neural network which is fully trained by backpropagation (AutoGrad).

* HNet is implemented in **/models/hclassifier.py**, which contains the code for the hierarchical probabilistic neural network model, as well as the RBOP
inference algorithm.

* FNet is implemented in **/models/fclassifier.py**, which contains the 
code for the flat probabilistic neural network model, as well as the UBOP 
inference algorithm.

## Running code 

### Training HNet and FNet

You can train HNet and FNet as follows:

```
export CUDA_VISIBLE_DEVICES=0 # use GPU

CUDA_VISIBLE_DEVICES=0 python3 -u train.py -pcsv ./data/VOC2006/TRAINVAL.csv -m hierarchical -s $( cat ./data/VOC2006/hierarchy_full.txt ) --gpu --vgg --no-ft 

CUDA_VISIBLE_DEVICES=0 python3 -u train.py -pcsv ./data/VOC2006/TRAINVAL.csv -m flat --gpu --vgg --no-ft 
```

### Testing HNet and FNet

Subsequently, you can test HNet and FNet as follows:

```
export CUDA_VISIBLE_DEVICES=0 # use GPU

CUDA_VISIBLE_DEVICES=0 python3 -u test.py -m flat -pm ./FLAT_MCC_TRAIN_0.2_True_100_0.001_32_5_False_True_VOC2006_VOC2006.pt -pcsv ./data/VOC2006/TEST.csv --gpu --vgg --store 

CUDA_VISIBLE_DEVICES=0 python3 -u test.py -m hierarchical -pm ./HIERARCHICAL_MCC_TRAIN_0.2_True_100_0.001_32_5_False_True_VOC2006_19_False_0_VOC2006.pt -pcsv ./data/VOC2006/TEST.csv -s $( cat ./data/VOC2006/hierarchy_full.txt ) --gpu --vgg --store 
```

*Important*: make sure to first train HNet and FNet before testing, as trained models need to be provided. After training, trained models are automatically saved to the working directory. 
