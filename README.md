# NTIRE2017

## Introduction
This repository is implemented for [NTIRE2017 Challenge](http://www.vision.ee.ethz.ch/ntire17/), based on [Facebook ResNet](https://github.com/facebook/fb.resnet.torch) and [SR ResNet](https://arxiv.org/pdf/1609.04802.pdf)

By [SNU-CVLAB](http://cv.snu.ac.kr/?page_id=57) Members; Seungjun Nah, Bee Lim, Heewon Kim, Sanghyun Son
## Model
This is our baseline model for scale 2. We only changed upsampler for different scale model.

![model_baseline x2](/document/model_baseline.png)

![model_upsamplers](/document/model_upsampler.png)

Multiscale model has three upsamplers for different scale inputs.

![model_multiscale](/document/model_multiscale.png)

## Challenge Results
Statistics | Individual Models| Multiscale Model| Leaderboard No.(IM/MSM) 
-- | -- | -- | --
Bicubic X2 |  |  | 
Bicubic X3 |  |  | 
Bicubic X4 |  |  | 
Unknown X2 |  |  | 
Unknown X3 |  |  | 
Unknown X4 |  |  | 

## Dependencies
* torch7
* cudnn
* nccl (Optional, for faster GPU communication)

## Code
Clone this repository into $makeReposit:
  ```bash
  makeReposit = /home/LBNet/
  mkdir -p $makeReposit/; cd $makeReposit/
  git clone https://github.com/LimBee/NTIRE2017.git
  ```

## Dataset
Please download the dataset from below.
* DIV2K produced by NTIRE2017
  ```bash
  makeData = /var/tmp/dataset/ # set absolute path as desired
  mkdir -p $makeData/; cd $makedata/
  wget https://cv.snu.ac.kr/~/DIV2K.tar
  tar -xvf DIV2K.tar
  ```
* Flickr2K collected by Flickr
  ```bash
  makeData = /var/tmp/dataset/ # set absolute path as desired
  mkdir -p $makeData/; cd $makedata/
  wget https://cv.snu.ac.kr/~/Flickr2K.tar
  tar -xvf Flickr2K.tar
  ```

Please convert the downloaded dataset into .t7 files (Recommended.)
* To train DIV2K
  ```bash
  cd $makeReposit/NTIRE2017/code/tools

  #This command generates multiple t7 files for
  #each images in DIV2K_train_HR folder
  th png_to_t7.lua -apath $makeData -dataset DIV2K -split true

  #This command generates a single t7 file that contains
  #every image in DIV2K_train_HR folder (Requires ~16GB RAM for training)
  th png_to_t7.lua -apath $makeData -dataset DIV2K -split false
  ```
* To train Flickr2K
  ```bash
  cd makeReposit/NTIRE2017/code/tools

  #This command generates multiple t7 files for
  #each images in Flickr2K_HR folder
  th png_to_t7.lua -apath $makeData -dataset Flickr2K -split true
  ```

Or, you can use just .png files. (Not recommended.) Details are described below
## Quick Start(Demo)
You can download our pre-trained models for each scale / degrader, and super-resolve your own image with our code.
1. Download pre-trained Individual and MultiScale models:
  ```bash
  cd $makeReposit/NTIRE2017/demo/model/
  wget https://cv.snu.ac.kr/~/bicubic_x2.t7
  wget https://cv.snu.ac.kr/~/bicubic_x3.t7
  wget https://cv.snu.ac.kr/~/bicubic_x4.t7
  wget https://cv.snu.ac.kr/~/unknown_x2.t7
  wget https://cv.snu.ac.kr/~/unknown_x3.t7
  wget https://cv.snu.ac.kr/~/unknown_x4.t7
  wget https://cv.snu.ac.kr/~/bicubic_mutiscale.t7
  ```
2. Run `test.lua` with a given model for selected validation image:
  ```bash
  cd $makeReposit/NTIRE2017/demo/
  # for bicubic x2
  th test.lua -type test -model bicubic_x2.t7 -scale 2 -nGPU 2 -selfEnsemble true -chopShave 10 -chopSize 16e4
  
  # for multiscale model
  th test.lua -type test -model multiscale.t7 -degrade bicubic -scale 2 -swap 1 -nGPU 2 -selfEnsemble true -chopShave 20 -chopSize 20e4 -dataDir ../../
  th test.lua -type test -model multiscale.t7 -degrade bicubic -scale 3 -swap 2 -nGPU 2 -selfEnsemble true -chopShave 20 -chopSize 24e4 -dataDir ../../
  th test.lua -type test -model multiscale.t7 -degrade bicubic -scale 4 -swap 3 -nGPU 2 -selfEnsemble true -chopShave 20 -chopSize 24e4 -dataDir ../../
  ```

## Training

1. To train default setting:
  ```bash
  th main.lua
  ```

2. To train fanal setting for individual bicubic x2:
  ```bash
  th main.lua -nFeat 256 -nResBlock 36 -patchSize 96 -scaleRes 0.1
  ```

3. To train fanal setting for multiScale model:
  ```bash
  th main.lua -nFeat 256 -nResBlock 36 -patchSize 96 -scaleRes 0.1
  ```
