# NTIRE2017: SNU_CVLab

# Introduction
This repository is implemented for [NTIRE2017 Challenge](http://www.vision.ee.ethz.ch/ntire17/), based on [Facebook ResNet](https://github.com/facebook/fb.resnet.torch) and [SR ResNet](https://arxiv.org/pdf/1609.04802.pdf)

by [SNU_CVLab Members](http://cv.snu.ac.kr/?page_id=57): **Seungjun Nah, Bee Lim, Heewon Kim, Sanghyun Son**
## Model
This is our **baseline** model for scale 2. We only changed upsampler for different scale model.

![model_baseline](/document/figs/baseline.png)

**Bicubic multiscale** model has three upsamplers to generate different scale output images.

![model_bicubic_multiscale](/document/figs/multiscale_bicubic.png)

**Unknown multiscale** model has three additional pre-processing modules for different scale inputs.

![model_unknown_multiscale](/document/figs/multiscale_unknown.png)

Every convolution layer execpt pre-processing modules in **Unknown multiscale** model uses **3x3** convolution kernel with **stride = 1, pad =  1**.

Each pre-processing module has two residual blocks with convolution kernel **5x5, stride = 1, pad = 1**.

## Challenge Results

To be announced

<!--
Model / PSNR (dB) | Expert | Multiscale | Ranking (Expert / Multiscale)
-- | -- | -- | --
Bicubic scale 2 |  |  | 
Bicubic scale 3 |  |  | 
Bicubic scale 4 |  |  | 
Unknown scale 2 |  |  | 
Unknown scale 3 |  |  | 
Unknown scale 4 |  |  | -->

<!--## Qualitative Results
Model | Input(bicubic interpolated) | Expert | Multiscale | Ground truth
-- | -- | -- | -- | --
Bicubic scale 2 |
Bicubic scale 3 |
Bicubic scale 4 |
Unknown scale 2 |
Unknown scale 3 |
Unknown scale 4 |-->

  <!--* We did not used above image (0791.png ~ 0800.png from **DIV2K** dataset) for training-->
<!--## Benchmark Results-->

# About our code
## Dependencies
* torch7
* cudnn
* nccl (Optional, for faster GPU communication)

## Code
Clone this repository into $makeReposit:
  ```bash
  makeReposit = [/the/directory/as/you/wish]
  mkdir -p $makeReposit/; cd $makeReposit/
  git clone https://github.com/LimBee/NTIRE2017.git
  ```

## Dataset
Please download the dataset from below.
* **DIV2K** provided by **NTIRE2017**
  ```bash
  makeData = /var/tmp/dataset/ # Please set the absolute path as desired
  mkdir -p $makeData/; cd $makedata/
  # Please download the dataset from [CodaLab](https://competitions.codalab.org/competitions/16303)
  tar -xvf DIV2K.tar
  ```
* **Flickr2K** collected by **SNU_CVLab** with Flickr API
  ```bash
  makeData = /var/tmp/dataset/
  mkdir -p $makeData/; cd $makedata/
  # Not prepared yet on the server cv.snu.ac.kr. (in April 18, 2017)
  # In a few days, we will release the code used to collect the images of Flickr2K. 
  wget https://cv.snu.ac.kr/~/Flickr2K.tar
  tar -xvf Flickr2K.tar
  ```
If you want to download the dataset into other location, please refer to the optional argument -dataset in opt.lua

To enable faster data loading, please convert the downloaded dataset into .t7 files
* To train with **DIV2K**
  ```bash
  cd $makeReposit/NTIRE2017/code/tools

  #This command generates multiple t7 files for
  #each images in DIV2K_train_HR folder
  th png_to_t7.lua -apath $makeData -dataset DIV2K -split true

  # This command generates a single t7 file that contains
  # every image in DIV2K_train_HR folder (Requires ~16GB RAM for training)
  th png_to_t7.lua -apath $makeData -dataset DIV2K -split false
  ```
* To train with **Flickr2K**
  ```bash
  cd $makeReposit/NTIRE2017/code/tools

  # This command generates multiple t7 files for
  # each images in Flickr2K_HR folder
  th png_to_t7.lua -apath $makeData -dataset Flickr2K -split true
  ```
You can choose to use .png filesm too. Details are described below.

## Quick Start (Demo)
You can download our pre-trained models and super-resolve your own image.

1. Download our models ([google drive link](https://drive.google.com/open?id=0B3AjYlPQo4LLR1FQOXdWTUhlSm8)) into $makeReposit/NTIRE2017/demo/model/

    A. Single scale-expert model

    * bicubic_x2.t7
    * bicubic_x3.t7
    * bicubic_x4.t7
    * unknown_x2_1.t7
    * unknown_x2_2.t7
    * unknown_x3_1.t7
    * unknown_x3_2.t7 
    * unknown_x4_1.t7
    * unknown_x4_2.t7

    (Below are the models that have been cut off at the deadline of the NTIRE2017 SR Challange and failed to upload. These perform slightly better than the model above.)
    * unknown_x3_3.t7
    * unknown_x3_4.t7 
    
    B. Multi-scale model
    * bicubic_multiscale.t7
    * unknown_multiscale_1.t7
    * unknown_multiscale_2.t7

<!--1. Download pre-trained **single-scale** and **multi-scale** models:
    ```bash
    cd $makeReposit/NTIRE2017/demo/model/

    # Bicubic single-scale 2
    wget https://cv.snu.ac.kr/NTIRE2017/model/bicubic_x2.t7
    # Bicubic single-scale 3
    wget https://cv.snu.ac.kr/NTIRE2017/model/bicubic_x3.t7
    # Bicubic single-scale 4
    wget https://cv.snu.ac.kr/NTIRE2017/model/bicubic_x4.t7
    
    # Unknown single-scale 2
    wget https://cv.snu.ac.kr/NTIRE2017/model/unknown_x2.t7
    # Unknown single-scale 3
    wget https://cv.snu.ac.kr/NTIRE2017/model/unknown_x3.t7
    # Unknown single-scale 4
    wget https://cv.snu.ac.kr/NTIRE2017/model/unknown_x4.t7
    
    # Bicubic multi-scale
    wget https://cv.snu.ac.kr/~/bicubic_multi.t7
    # Unknown multi-scale
    wget https://cv.snu.ac.kr/~/bicubic_multi.t7
    ```-->

2. Run `test.lua` with given models and images:
    You can reproduce our final results with command below.

    ```bash
    cd $makeReposit/NTIRE2017/demo/

    # Bicubic scale 2
    th test.lua -type test -model bicubic_x2.t7 -scale 2 -selfEnsemble true
    # Bicubic scale 3
    th test.lua -type test -model bicubic_x3.t7 -scale 3 -selfEnsemble true
    # Bicubic scale 4
    th test.lua -type test -model bicubic_x4.t7 -scale 4 -selfEnsemble true

    # Unknown scale 2
    th test.lua -type test -model unknown_x2_1.t7+unknown_x2_2.t7 -scale 2 -degrade unknown -selfEnsemble false
    # Unknown scale 3
    th test.lua -type test -model unknown_x3_1.t7+unknown_x3_2.t7 -scale 3 -degrade unknown -selfEnsemble false
    # Unknown scale 4
    th test.lua -type test -model unknown_x4_1.t7+unknown+x4_2.t7 -scale 4 -degrade unknown -selfEnsemble false

    # Bicubic multiscale (Note that scale 2, 3, 4 share the same model!)
    th test.lua -type test -model bicubic_multiscale.t7 -degrade bicubic -scale 2 -selfEnsemble true
    th test.lua -type test -model bicubic_multiscale.t7 -degrade bicubic -scale 3 -selfEnsemble true
    th test.lua -type test -model bicubic_multiscale.t7 -degrade bicubic -scale 4 -selfEnsemble true

    # Unknown multiscale (Note that scale 2, 3, 4 share the same model!)
    th test.lua -type test -model unknown_multiscale_1.t7+unknown_multiscale_2.t7 -degrade unknown -scale 2 -selfEnsemble false
    th test.lua -type test -model unknown_multiscale_1.t7+unknown_multiscale_2.t7 -degrade unknown -scale 3 -selfEnsemble false
    th test.lua -type test -model unknown_multiscale_1.t7+unknown_multiscale_2.t7 -degrade unknown -scale 4 -selfEnsemble false
    ```
        
    * Here are some optional arguments
      ```bash
      -nGPU     [n]   # You can test our model with multiple GPU. (n = 1, 2, 4)

      -dataDir  [$makeData]           #Please specify this directory. Default is /var/tmp/dataset
      -type     [bench | test | val]
      -dataset  [DIV2K | myData]
      -save     [Folder name]

      -selfEnsemble [true | false]    # Do not use this option for unknown downsampling.
      
      -chopSize [S]   # Please reduce the chopSize when test fails due to GPU memory.
                      # The optimal size of S can be vary depend on your maximum GPU memory.

      -progress [true | false]
      ```
    Or, you can test with your own model and images.
    ```bash
    th test.lua -type test -dataset myData -model anyModel -scale [2 | 3 | 4] -degrade [bicubic | unknown]
    ```
    This code generates high-resolution images for some famous SR benchmark set (Set 5, Set 14, Urban 100, BSD 100)
    ```bash
    th test.lua -type bench -model anyModel -scale [2 | 3 | 4]
    ```
    We used 0791.png to 0800.png in DIV2K train set for validation, and you can test any model with validation set.
    ```bash
    th test.lua -type val -model anyModel -scale [2 | 3 | 4] -degrade [bicubic | unknown]
    ```
    If you have ground-truth images for the test images, you can evaluate them with MATLAB. (-type [bench | val] automatically place ground-truth high-resolution images into img_target folder.)
    ```bash
    matlab -nodisplay <evaluation.m
    ```

## Training

* To train baseline model:
    ```bash
    th main.lua
    ```
 <!---
  * Here are some optional arguments. Please specify them as you wish. (Recommended)
      ```bash
      -nGPU     [n] # You can train expert model with multiple GPU. (Not multiscale model.)
      -nThreads [n] # Number of threads for data loading.

      -datadir [$makeData]  # Please specify this directory. Default is /var/tmp/dataset

      -save [Folder name]   # You can generate experiment folder with given name.
      -load [Folder name]   # You can resume your experiment from the last checkpoint.
                            # Please do not set -save and -load at the same time.

      -nEpochs    [n]                   # Number of epochs to run
      -testEvery  [n]                   # Iterations per one epoch
      -datatype   [png | t7 | t7pack]   # png < t7 < t7pack - requires larger memory
                                        # png > t7 > t7pack - requires faster CPU & Storage

      -chopSize   [S]   # Please reduce the chopSize when test fails due to GPU memory.
                        # The optimal size of S can be vary depend on your maximum GPU memory.
      ```
      --->
* To train bicubic model :
    ```bash
    # Bicubic scale 2 from scratch
    th main.lua -scale 2 -nFeat 256 -nResBlock 36 -patchSize 96 -scaleRes 0.1
    # Bicubic scale 3 from pretrained bicubic scale 2 model
    th main.lua -scale 3 -nFeat 256 -nResBlock 36 -patchSize 144 -scaleRes 0.1 -preTrained [Bicubic scale 2]
    # Bicubic scale 4 from pretrained bicubic scale 2 model
    th main.lua -scale 4 -nFeat 256 -nResBlock 36 -patchSize 192 -scaleRes 0.1 -preTrained [Bicubic scale 2]
    ```

* To train unknown model from scratch:
    ```bash
    # Unknown scale 2 from pretrained bicubic scale 2 model
    th main.lua -scale 2 -degrade unknown -nFeat 256 -nResBlock 36 -patchSize 96 -scaleRes 0.1 -preTrained [Bicubic scale 2]
    # Unknown scale 3 from pretrained bicubic scale 2 model
    th main.lua -scale 3 -degrade unknown -nFeat 256 -nResBlock 36 -patchSize 144 -scaleRes 0.1 -preTrained [Bicubic scale 2]
    # Unknown scale 4 from pretrained bicubic scale 2 model
    th main.lua -scale 4 -degrade unknown -nFeat 256 -nResBlock 36 -patchSize 192 -scaleRes 0.1 -preTrained [Bicubic scale 2]
    ```
<!---
* We used bicubic scale 2 pre-trained model to train the other models. To use pre-trained model, use -preTrained option.
    ```bash
    # Bicubic scale 3 using Bicubic scale 2 pre-trained model
    th main.lua -scale 3 -degrade bicubic -nFeat 256 -nResBlock 36
    -patchSize 96 -scaleRes 0.1 -preTrained [Bicubic scale 2 directory]

    # Unknown scale 3 using Bicubic scale 2 pre-trained model
    th main.lua -scale 3 -degrade unknown -nFeat 256 -nResBlock 36
    -patchSize 144 -scaleRes 0.1 -preTrained [Bicubic scale 2 directory]
    ```
    --->
* To train bicubic multiscale model:
    ```bash
    th main.lua -scale 2_3_4 -netType multiscale -nResBlock 80
    -patchSize 64 -multiPatch true -skipBatch 3
    ```
* We used bicubic multiscale pre-trained model to train the unknown multiscale model.
    ```bash
    th main.lua -scale 2_3_4 -netType multiscale_unknown -degrade unknown
    -preTrained [Bicubic multiscale directory] -multiPatch true -skipBatch 3
    ```

## Experiment
