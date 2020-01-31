# HD<sup>3</sup>

This is a PyTorch implementation of our paper:

Hierarchical Discrete Distribution Decomposition for Match Density Estimation (CVPR 2019)

[Zhichao Yin](http://zhichaoyin.me/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Fisher Yu](https://www.yf.io/)

We propose a framework suitable for learning probabilistic pixel correspondences. It has applications including but not limited to stereo matching and optical flow, with inherent uncertainty estimation. HD<sup>3</sup> achieves state-of-the-art results for both tasks on established benchmarks ([KITTI](http://www.cvlibs.net/datasets/kitti/index.php) & [MPI Sintel](http://sintel.is.tue.mpg.de/)).

arxiv preprint: (https://arxiv.org/abs/1812.06264)

<img src="misc/teaser.jpg" width="750">

## Requirement

This code has been tested with Python 3.6, PyTorch 1.0 and CUDA 9.0 on Ubuntu 16.04.

## Getting Started
- Install PyTorch 1.0 and we recommend using anaconda3 for managing the python environment. You can install all the dependencies by the following:
```bash
pip install -r requirements.txt
```
- Download all the relevant datasets including the [FlyingChairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html), the [FlyingThings3D dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (we use ``DispNet/FlowNet2.0 dataset subsets`` following the practice of FlowNet 2.0), the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/index.php), and the [MPI Sintel dataset](http://sintel.is.tue.mpg.de/).

## Model Training
To train a model on a specific dataset, simply run
```bash
bash scripts/train.sh
```
Note the scripts contain several placeholders which you should replace with your customized choices. For instance, you can specify the dataset type (e.g. FlyingChairs) via `--dataset_name`, alternate the network architecture via `--encoder` and `--decoder`, and switch the task (stereo or flow) you solve via `--task`. You can also partly load the weights of a pretrained backbone network via `--pretrain_base` (download ImageNet pretrained DLA-34 [here](http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth)), or strictly initialize the weights from a pretrained model via `--pretrain`.

You can then start a tensorboard session by
```bash
tensorboard --logdir=/path/to/log/files --port=8964
```
and visualize your training progress by accessing [https://localhost:8964](https://localhost:8964) on you browser.
- We provide the learning rate schedules and augmentation configurations in all of our experiments. For other detailed hyperparameters, please refer to our paper so as to reproduce our result.

## Model Inference
To test a model on a folder of images, please run
```bash
bash scripts/test.sh
```
Please provide the list of image pair names and pass it to `--data_list`. This script will generate predictions for every pair of images and save them in the `--save_folder` with the same folder hierarchy as input images. You can choose the saved flow format (e.g. png or flo) via `--flow_format`. When the folder contains images of different input sizes (e.g. KITTI), please make sure the `--batch_size` is 1.
- When the ground truth is available, you can optionally enable the argument `--evaluate` to calculate the End-Point-Error of your predictions. Please make sure the list consists of `img-name1 img-name2 gtruth-name` in each line.

## Model Zoo
We provide pretrained models for all of our experiments. To download them, simply run
```bash
bash scripts/download_models.sh
```
The names of the models come in the format of `model-name_dataset-names`. Models are named as `hd3f/hd3s` for optical flow and stereo matching. A suffix of `c` is appended for models with context module. The `dataset_names` indicates our dataset schedule for training the model. You should be able to obtain similar results by running the test script we provide.

## Citation
If you find our work or our repo useful in your research, please consider citing our paper:
```
@InProceedings{Yin_2019_CVPR,
author = {Yin, Zhichao and Darrell, Trevor and Yu, Fisher},
title = {Hierarchical Discrete Distribution Decomposition for Match Density Estimation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## FAQ
- Why are the model outputs different even for the same input in different runs?

  Some PyTorch ops are non-deterministic (e.g. torch.tensor.scatter_add_). If you fix all the random seeds for Python and PyTorch, you shall get identical results.
  
- Why does the model finetuned on the KITTI dataset exhibit artifacts in the sky regions?

  This is due to the limited amount of data during finetuning stage. Effective solutions to resolve it include an additional smoothness loss term during finetuning and knowledge distillation from the model pretrained on the synthetic datasets.

## Acknowledgements
We thank [Houning Hu](https://eborboihuc.github.io/) for making the [teaser image](https://github.com/ucbdrive/hd3/blob/master/misc/teaser.jpg), [Simon Niklaus](http://sniklaus.com/) for the [correlation operator](https://github.com/sniklaus/pytorch-pwc) and [Clément Pinard](http://perso.ensta.fr/~pinard/) for the [FlowNet implementation](https://github.com/ClementPinard/FlowNetPytorch).
