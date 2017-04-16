# NTIRE2017

## Introduction
This repository is implemented for [NTIRE2017 Challenge](http://www.vision.ee.ethz.ch/ntire17/), based on [FaceBook ResNet](https://github.com/facebook/fb.resnet.torch) and [SR ResNet](https://arxiv.org/pdf/1609.04802.pdf)

By [SNU-CVLAB](http://cv.snu.ac.kr/) Members, Lim Bee, Sanghyun Son, Seungjun Nah, Heewon Kim

## Challenge Results
Statistics | Individual Models| MultiScale Model| Leaderboard No.(IM/MSM) 
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

## Dataset
Download Dataset
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
   

## Quick Start
To run pretrained DeepMask/SharpMask models to generate object proposals, follow these steps:

1. Download Dataset into $makeData:

2. Clone this repository into $makeReposit:

   ```bash
   makeReposit = /home/LBNet/
   mkdir -p $makeReposit/; cd $makeReposit/
   git clone https://github.com/LimBee/NTIRE2017.git
   ```

3. Download pre-trained Individual and MultiScale models:

   ```bash
   mkdir -p $makeReposit/NTIRE2017/demo/model/; cd $makeReposit/NTIRE2017/demo/model/
   wget https://cv.snu.ac.kr/~/bicubic_x2.t7 # Individual (bicubic_x3.t7 ~ unknown_x4.t7) 
   wget https://cv.snu.ac.kr/~/multiScale_model.t7
   ```

3. Run `computeProposals.lua` with a given model and optional target image (specified via the `-img` option):

   ```bash
   # apply to a default sample image (data/testImage.jpg)
   cd $DEEPMASK
   th computeProposals.lua $DEEPMASK/pretrained/deepmask # run DeepMask
   th computeProposals.lua $DEEPMASK/pretrained/sharpmask # run SharpMask
   th computeProposals.lua $DEEPMASK/pretrained/sharpmask -img /path/to/image.jpg
   ```

## Training



## Demo
