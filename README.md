# HD<sup>3</sup>

This is a PyTorch implementation of our paper:

Hierarchical Discrete Distribution Decomposition for Match Density Estimation (CVPR 2019)

[Zhichao Yin](http://zhichaoyin.me/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Fisher Yu](https://www.yf.io/)

It tackles the problem of probabilistic pixel matching, with applications including stereo matching and optical flow. HD<sup>3</sup> achieves state-of-the-art results for both tasks on established benchmarks ([KITTI](http://www.cvlibs.net/datasets/kitti/index.php) & [MPI Sintel](http://sintel.is.tue.mpg.de/)).

arxiv preprint: (https://arxiv.org/abs/1812.06264)

<img src="misc/teaser.jpg" width="750">

## Requirement

This code has been tested with Python 3.6, PyTorch 1.0 and CUDA 9.0 on Ubuntu 16.04.

## Getting Started
- Clone this repo:
```bash
git clone https://github.com/ucbdrive/hd3
cd hd3
```
- Install PyTorch 1.0 and we recommend using anaconda3 for managing the python environment. You can install all the dependencies by the following:
```bash
pip install -r requirements.txt
```
- Download all the relevant datasets including the [FlyingChairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html), the [FlyingThings3D dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (we use ``DispNet/FlowNet2.0 dataset subsets`` following the practice of FlowNet 2.0), the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/index.php), and the [MPI Sintel dataset](http://sintel.is.tue.mpg.de/).

## Model training
To train a model on a specific dataset, simply run
```bash
bash scripts/train.sh
```
You can specify the dataset type (e.g. FlyingChairs) via `--dataset_name`, alternate the network architecture via `--encoder` and `--decoder`, and switch the task (stereo or flow) you solve via `--task`.
- We provide the learning rate schedules and augmentation configurations in all of our experiments. For other detailed hyperparameters, please refer to our paper so as to reproduce our result.

## Model inference
To test a model on a folder of images, please run
```bash
bash scripts/test.sh
```
Please provide the list of image pair names and pass it to `--data_list`. This script will generate predictions for every pair of images and save them in the `--save_folder` with the same folder hierarchy as input images. You can choose the saved flow format (e.g. png or flo) via `--flow_format`.

## Citation
If you find our work or our repo useful in your research, please consider citing our paper:
```
@inproceedings{yin2019hd3,
title = {Hierarchical Discrete Distribution Decomposition
for Match Density Estimation},
author = {Yin, Zhichao and Darrell, Trevor and Yu,
Fisher},
booktitle = {CVPR},
year = {2019}
}
```

## Acknowledgements
We thank [Simon Niklaus](http://sniklaus.com/) for his PyTorch implementation of the [correlation operator](https://github.com/sniklaus/pytorch-pwc) and [Clément Pinard](http://perso.ensta.fr/~pinard/) for his PyTorch [FlowNet implementation](https://github.com/ClementPinard/FlowNetPytorch).
