# pytorch-SelectiveNet

This repo is adapted from the [unofficial pytorch implementation](https://github.com/gatheluck/pytorch-SelectiveNet) of paper "SelectiveNet: A Deep Neural Network with an Integrated Reject Option" [Geifman+, ICML2019].

See original repo for requirements and references [here](https://github.com/gatheluck/pytorch-SelectiveNet)


Below are instructions on how to reproduce our CIFAR-10 and ImageNet results used in our paper.

### Training the ScNet
Use `scripts/train.py` to train the CIFAR-10 network:
```bash
# Example usage
cd scripts
python train.py --dataset cifar10 --log_dir ../logs/train --coverage 0.7 --data_root <YOUR_DIR_TO_CIFAR10_DATASET>
```

Use `scripts/train_imagenet.py` to train the CIFAR-10 network:
```bash
# Example usage
cd scripts
python train_imagenet.py --dataset imagenet --log_dir ../logs/train_imgnet --coverage 0.7 --num_workers 16 --batch_size 768 --num_epochs 250 --lr 0.1 --dataroot <YOUR_DIR_TO_IMAGENET_DATASET>
```

### Papare data for selective classification analysis

* Please first edit the file "collect_logits.py" --- fill in all dataset paths to the variable "parser". This will bring a lot convenience to the run later.
* Below are the list (and url reference) of datasets you will need to prepare before running the experiments:

    [ImageNet](https://www.image-net.org/): at least the validation set (of 2012 version) and the meta file are needed for selective classfication analysis.

    [ImageNet-C](https://zenodo.org/records/2235448): at leaset prepare all 19 types of corruptions with severity 3 to reproduce the result in the paper.

    [ImageNet-O](https://github.com/hendrycks/natural-adv-examples)

    [OpenImage-O](https://ooddetection.github.io/): Note that user only needs to download OpenImage V3; the data split file "openimage_o.txt" has been attached in this repo at "./openimage_o.txt".

    [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

    [CIFAR-10-C](https://github.com/hendrycks/robustness)

```bash
# This will be sufficient to run all logits collection for ImageNet SC model.
python collect_logits.py --dataset imagenet-c 

# This will be sufficient to run all logits collection for CIFAR SC model.
python collect_logits.py --dataset cifar10-c 

```