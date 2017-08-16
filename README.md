# NTIRE2017 Super-resolution Challenge: SNU_CVLab

<!--## Introduction-->
This is a repository for **Team SNU_CVLab**,  the winner of [NTIRE2017 Challenge on Single Image Super-Resolution](http://www.vision.ee.ethz.ch/ntire17/).<br><br>
Our paper ***"Enhanced Deep Residual Networks for Single Image Super-Resolution"*** [(PDF)](http://cv.snu.ac.kr/publication/conf/2017/EDSR_fixed.pdf) won the **Best Paper Award** of the [NTIRE2017 workshop](http://www.vision.ee.ethz.ch/ntire17/) from CVPR2017.

Team members:<br>
**Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee** from [Computer Vision Lab, SNU](http://cv.snu.ac.kr/?page_id=19)<br><br>
The codes are based on [Facebook ResNet](https://github.com/facebook/fb.resnet.torch). <br>

## Model
**EDSR** (Single-scale model. We provide scale x2, x3, x4 model).

![EDSR](/figs/EDSR.png)

**MDSR** (Multi-scale model. It can handle x2, x3, x4 super-resolution with a single model).

![MDSR](/figs/MDSR.png)

Please see our [paper](http://cv.snu.ac.kr/publication/conf/2017/EDSR_fixed.pdf) for more details.

## NTIRE2017 Super-resolution Challenge Results

Our team (**SNU_CVLab**) won the 1st (EDSR) and 2nd (MDSR) prize.

![Challenge_result](/figs/Challenge_result.png)


# About our code
## Dependencies
* torch7
* cudnn
* nccl (Optional, for faster GPU communication)

## Code
Clone this repository into any place you want. You may follow the example below.
```bash
makeReposit = [/the/directory/as/you/wish]
mkdir -p $makeReposit/; cd $makeReposit/
git clone https://github.com/LimBee/NTIRE2017.git
```

## Dataset
Please download the dataset from [here](http://cv.snu.ac.kr/research/EDSR/DIV2K.tar) if you want to train our models from scratch or evaluate the DIV2K dataset. Place the tar file anywhere you want. **(We recommend /var/tmp/dataset/DIV2K.tar)** Then, please follow the guide below. <U>If want to place the dataset in the other directory, **you have to change the optional argument -dataset for training and test.**</U>
* **DIV2K** from [**NTIRE2017**](http://www.vision.ee.ethz.ch/ntire17/)
    ```bash
    makeData = /var/tmp/dataset/ # We recommend this path, but you can freely change it.
    mkdir -p $makeData/; cd $makedata/
    tar -xvf DIV2K.tar
    ```
    You should have the following directory structure:

    `/var/tmp/dataset/DIV2K/DIV2K_train_HR/0???.png`<br>
    `/var/tmp/dataset/DIV2K/DIV2K_train_LR_bicubic/X?/0???.png`<br>
    `/var/tmp/dataset/DIV2K/DIV2K_train_LR_unknown/X?/0???.png`<br>

* **Flickr2K** collected by us using Flickr API
    ```bash
    makeData = /var/tmp/dataset/
    mkdir -p $makeData/; cd $makedata/
    wget http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
    tar -xvf Flickr2K.tar
    ```
    You should have the following directory structure:

    `/var/tmp/dataset/Flickr2K/Flickr2K_HR/00????.png`<br>
    `/var/tmp/dataset/Flickr2K/Flickr2K_train_LR_bicubic/X?/00????x?.png`<br>
    `/var/tmp/dataset/Flickr2K/Flickr2K_train_LR_unknown/X?/00????x?.png`<br>

    To generate the training images for unknown track, we trained simple downsampler network.<br>
    You can download them from [here](htt://cv.snu.ac.kr/research/EDSR/downsamplers.tar).

To make data loading faster, you can convert the dataset into binary .t7 files
* Convert **DIV2K** dataset into .t7 files
    ```bash
    cd $makeReposit/NTIRE2017/code/tools

    # Choose one

    # This command generates multiple t7 files for
    # each image in DIV2K_train_HR folder (Requires ~2GB RAM for training)
    th png_to_t7.lua -apath $makeData -dataset DIV2K -split true

    # This command generates a single t7 file that contains
    # every image in DIV2K_train_HR folder (Requires ~16GB RAM for training)
    th png_to_t7.lua -apath $makeData -dataset DIV2K -split false
    ```
* Convert **Flickr2K** dataset into .t7 files
    ```bash
    cd $makeReposit/NTIRE2017/code/tools

    # This command generates multiple t7 files for
    # each image in Flickr2K_HR folder
    th png_to_t7.lua -apath $makeData -dataset Flickr2K -split true
    ```
You can use raw .png files too. Please see **Training** for the details.

## Quick Start (Demo)
You can download our pre-trained models upscale your own image.

1. Download our models using the scripts below. You can also download them from [here](http://cv.snu.ac.kr/research/EDSR/model_paper.tar).After download the tar file, make sure  that the file is placed in the right directory. (`$makeReposit/NTIRE2017/demo/model/model_paper.tar`)<br>
**We recommend you to download the models for paper because the models for challenge is not compatible with our current code. Please contact us if you want to execute those models.**
    ```bash
    cd $makeReposit/NTIRE2017/demo/model/

    # Our models submitted to the Challenge
    wget http://cv.snu.ac.kr/research/EDSR/model_challenge.tar

    # Our models for the paper
    wget http://cv.snu.ac.kr/research/EDSR/model_paper.tar
    ```

2. Run `test.lua` with given models and images:
    
    ```bash
    cd $makeReposit/NTIRE2017/demo

    # This command runs our EDSR+ (scale 2)
    th test.lua -selfEnsemble true

    # This command runs our MDSR+ (scale 2)
    th test.lua -model bicubic_multiscale -scale 2 -selfEnsemble true
    ```
    You can reproduce our final results with `makeFinal.sh` in `NTIRE2017/demo` directory. You have to uncomment the line you want to execute.
    ```bash
    sh makeFinal.sh
    ```

    * Here are some optional arguments you can adjust. If you have any problem, please refer following lines.

        ```bash
        # You can test our model with multiple GPU. (n = 1, 2, 4)
        -nGPU       [n]

        # Please specify this directory. Default is /var/tmp/dataset
        -dataDir    [$makeData]
        -dataset    [DIV2K | myData]
        -save       [Folder name]

        # About self-ensemble strategy, please see our paper for the detail.
        # Do not use this option for unknown downsampling.
        -selfEnsemble   [true | false]

        # Please reduce the chopSize when you see 'out of memory'.
        # The optimal size of S can be vary depend on your maximum GPU memory.
        -chopSize   [S]   
        ```

        For some reasons, our torch code does not support full evaluation. You have to run **MATLAB** script to get the output PSNR and SSIM.

        ```bash
        matlab -nodisplay <evaluation.m
        ```

        If you do not want to calculate SSIM, please modify `evaluation.m` file as below. (Calculating SSIM of large image is very slow.)
        ```
        line 6:     psnrOnly = false; -> psnrOnly = true;
        ```
    You can run the test script with your own model and images. Just put your images in `NTIRE2017/demo/img_input`. If you have ground-truth high-resolution images, please locate them in **NTIRE2017/demo/img_target/myData** for evaluation.
    
    ```bash
    th test.lua -type test -dataset myData -model anyModel -scale [2 | 3 | 4] -degrade [bicubic | unknown]
    ```
    To run the **MDSR**, model name should include `multiscale`. (For example, `multiscale_blahblahblah.t7`) Sorry for inconvinience.
    <!---
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
    --->
## Training

1. To train our baseline model, please run the following command:

    ```bash
    th main.lua         # This model is not our final model!
    ```

    * Here are some optional arguments you can adjust. If you have any problem, please refer following lines. You can check more information in `NTIRE2017/code/opts.lua`.
        ```bash
        # You can train the model with multiple GPU. (Not multi-scale model.)
        -nGPU       [n]

        # Number of threads for data loading.
        -nThreads   [n]   

        # Please specify this directory. Default is /var/tmp/dataset
        -datadir    [$makeData]  

        # You can make an experiment folder with the name you want.
        -save       [Folder name]

        # You can resume your experiment from the last checkpoint.
        # Please do not set -save and -load at the same time.
        -load       [Folder name]     

        # png < t7 < t7pack - requires larger memory
        # png > t7 > t7pack - requires faster CPU & Storage
        -datatype   [png | t7 | t7pack]     

        # Please increase the splitBatch when you see 'out of memory' during training.
        # S should be the power of 2. (1, 2, 4, ...)
        -splitBatch [S]

        # Please reduce the chopSize when you see 'out of memory' during test.
        # The optimal size of S can be vary depend on your maximum GPU memory.
        -chopSize   [S]
        ```

2. To train our EDSR and MDSR, please use the `training.sh` in `NTIRE2017/code` directory. You have to uncomment the line you want to execute.

    ```bash
    cd $makeReposit/NTIRE2017/code
    sh training.sh
    ```

    <U>Some model may require pre-trained **bicubic scale 2** or **bicubic multiscale** model.</U> Here, we assume that you already downloaded `bicubic_x2.t7` and `bicubic_multiscale.t7` in the `NTIRE2017/demo/model` directory. Otherwise, you can create them yourself. It is possible to start the traning from scratch by removing `-preTrained` option in the script.