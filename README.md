# NTIRE2017 Super-resolution Challenge: SNU_CVLab

# Introduction
This repository is for [NTIRE2017 Challenge](http://www.vision.ee.ethz.ch/ntire17/), based on [Facebook ResNet](https://github.com/facebook/fb.resnet.torch) and [SR ResNet](https://arxiv.org/pdf/1609.04802.pdf)

by [SNU_CVLab Members](http://cv.snu.ac.kr/?page_id=57): **Seungjun Nah, Bee Lim, Heewon Kim, Sanghyun Son, KyoungMu Lee**

## Model
This is our **single scale** model for scale 2. We only changed upsampler for different scale models.

![model_baseline](/document/figs/baseline.png)

**Bicubic multiscale** model has three upsamplers to generate different scale output images.

![model_bicubic_multiscale](/document/figs/multiscale_bicubic.png)

**Unknown multiscale** model has three additional pre-processing modules for different scale inputs.

![model_unknown_multiscale](/document/figs/multiscale_unknown.png)

Every convolution layer execpt pre-processing modules in **Unknown multiscale** model uses **3x3** convolution kernel with **stride = 1, pad =  1**.

Each pre-processing module has two residual blocks with convolution kernel **5x5, stride = 1, pad = 1**.

## Challenge Results

To be announced

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
Please download the dataset from below. If you download the dataset in the other location, <U>you have to change the optional argument -dataset when running **NTIRE2017/code/main.lua** file.</U>
* **DIV2K** provided by **NTIRE2017**
    ```bash
    makeData = /var/tmp/dataset/ # Please set the absolute path as desired
    mkdir -p $makeData/; cd $makedata/
    # Please download the dataset from [CodaLab](https://competitions.codalab.org/competitions/16303)
    tar -xvf DIV2K.tar
    ```
    After untar, you will have the following directory structure:

    **/var/tmp/dataset/DIV2K/DIV2K_train_HR/0???.png**

    **/var/tmp/dataset/DIV2K/DIV2K_train_LR_bicubic/X?/0???.png**

    **/var/tmp/dataset/DIV2K/DIV2K_train_LR_unknown/X?/0???.png**

* **Flickr2K** collected by **SNU_CVLab** with Flickr API
    ```bash
    makeData = /var/tmp/dataset/
    mkdir -p $makeData/; cd $makedata/

    # Not prepared yet on the server cv.snu.ac.kr. (in April 18, 2017)
    # In a few days, we will release the code used to collect the images of Flickr2K. 
    wget https://cv.snu.ac.kr/~/Flickr2K.tar
    tar -xvf Flickr2K.tar
    ```

    After untar, you will have the following directory structure:

    **/var/tmp/dataset/Flickr2K/Flickr2K_HR/00????.png** 
    
    **/var/tmp/dataset/Flickr2K/Flickr2K_train_LR_bicubic/X?/00????x?.png**

    **/var/tmp/dataset/Flickr2K/Flickr2K_train_LR_unknown/X?/00????x?.png**

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

    **A. Single scale-expert model**
    
    **Track 1**
    * bicubic_x2.t7
    * bicubic_x3.t7
    * bicubic_x4.t7
    
    **Track 2**
    * unknown_x2_1.t7
    * unknown_x2_2.t7
    * unknown_x3_1.t7
    * unknown_x3_2.t7 
    * unknown_x4_1.t7
    * unknown_x4_2.t7

    (Below are the models that have been cut off at the deadline of the NTIRE2017 SR Challange and failed to upload. These perform slightly better than the model above.)
    * unknown_x3_3.t7
    * unknown_x3_4.t7 
    
    **B. Multi-scale model**
    
    **Track 1**
    * bicubic_multiscale.t7
    
    **Track 2**
    * unknown_multiscale_1.t7
    * unknown_multiscale_2.t7

2. Run `test.lua` with given models and images:
    
    ```bash
    cd $makeReposit/NTIRE2017/demo

    th test.lua -selfEnsemble true      # This command runs our final bicubic_x2 model
    ```
    You can reproduce our final results with provided shell script **makeFinal.sh** in **NTIRE2017/demo** directory. You have to uncomment the appropriate line before you run.
    ```bash
    sh makeFinal.sh
    ```

    * Here are some optional arguments you can adjust. If you have any problem in running above examples, please refer following line.

        ```bash
        -nGPU     [n]   # You can test our model with multiple GPU. (n = 1, 2, 4)

        -dataDir  [$makeData]           #Please specify this directory. Default is /var/tmp/dataset
        -type     [bench | test | val]
        -dataset  [DIV2K | myData]
        -save     [Folder name]

        -selfEnsemble [true | false]        # Generates 8 output images using single model.
                                            # Do not use this option for unknown downsampling.

        -chopSize [S]   # Please reduce the chopSize when test fails due to GPU memory.
                        # The optimal size of S can be vary depend on your maximum GPU memory.

        -progress [true | false]
        ```

        **-type val** uses down-sampled **DIV2K 0791~0800** to generate the output. Please note that we did not used those images for training. You can check how different options affect the total performance by runnning evaluation after you generated sets of validation images. (Requires MATLAB)

        ```bash
        matlab -nodisplay <evaluation.m
        ```

        If you do not want to calculate SSIM, please modify evaluation.m file as below. Calculating PSNR only will take a few seconds.
        ```
        line 6:     psnrOnly = false; -> psnrOnly = true;
        ```
    You can test with your own model and images. Just put your images in **NTIRE2017/demo/img_input** directory. If you have ground-truth high-resolution images for your own dataset, please locate them in **NTIRE2017/demo/img_target/myData** to evaluate the results.
    
    ```bash
    th test.lua -type test -dataset myData -model anyModel -scale [2 | 3 | 4] -degrade [bicubic | unknown]
    ```

    Soon, we will support some famous super-resolution benchmark sets like **Set5, Set14, Unban100, BSD100**.
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

1. To train our shallow baseline model, please run the following command:

    ```bash
    th main.lua         # This model is not our final model!
    ```

    * Here are some optional arguments you can adjust. If you have any problem in running above examples, please refer following line. For more information about arguments, please refer to **NTIRE2017/code/opts.lua** file.
        ```bash
        -nGPU     [n]   # You can train expert model with multiple GPU. (Not multiscale model.)
        -nThreads [n]   # Number of threads for data loading.

        -datadir [$makeData]  # Please specify this directory. Default is /var/tmp/dataset

        -save [Folder name]     # You can generate experiment folder with given name.
        -load [Folder name]     # You can resume your experiment from the last checkpoint.
                                # Please do not set -save and -load at the same time.

        -datatype   [png | t7 | t7pack]     # png < t7 < t7pack - requires larger memory
                                            # png > t7 > t7pack - requires faster CPU & Storage

        -chopSize   [S]     # Please reduce the chopSize when test fails due to GPU memory.
                            # The optimal size of S can be vary depend on your maximum GPU memory.
        ```

2. To train our final model, please use the provided shell script **training.sh** in **NTIRE2017/code** directory. You have to uncomment the appropriate line before you run. 

    <U>Some model requires pre-trained **bicubic scale 2** or **bicubic multiscale** model.</U> Here, we assume that you already downloaded the model **bicubic_x2.t7** and **bicubic_multiscale.t7** in the **NTIRE2017/demo/model** directory. It is possible to start the traning from scratch by removing -preTrained option in the script.

    ```bash
    cd $makeReposit/NTIRE2017/code

    sh training.sh
    ```
