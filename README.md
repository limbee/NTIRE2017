# NTIRE2017 Super-resolution Challenge: SNU_CVLab

## Introduction
This is our project repository for CVPR 2017 Workshop ([2nd NTIRE](http://www.vision.ee.ethz.ch/ntire17/)).

We, **Team SNU_CVLab**, (<i>Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah</i>, and <i>Kyoung Mu Lee</i> of [**Computer Vision Lab, Seoul National University**](http://cv.snu.ac.kr/)) are **winners** of [**NTIRE2017 Challenge on Single Image Super-Resolution**](http://www.vision.ee.ethz.ch/ntire17/). 

Our paper was published in CVPR 2017 workshop ([2nd NTIRE](http://www.vision.ee.ethz.ch/ntire17/)), and won the **Best Paper Award** of the workshop challenge track.




Please refer to our paper for details. 

If you use our work useful in your research or publication, please cite our work:

[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017** </i> [[PDF](http://cv.snu.ac.kr/publication/conf2017/EDSR_fixed.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] 
```
@inproceedings{lim2017enhanced,
  title={Enhanced Deep Residual Networks for Single Image Super-Resolution},
  author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month={July},
  year={2017}
}
```

In this repository, we provide
* Our model architecture description (EDSR, MDSR)
* NTIRE2017 Super-resolution Challenge Results
* Demo code
* Training code
* Download link of our trained models (EDSR, MDSR) 
* Download link of the datasets we used (DIV2K, Flickr2K)
* Super-resolution examples

The code is based on Facebook's Torch implementation of ResNet ([facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)). <br>

## Model Architecture
**EDSR** (Single-scale model. We provide scale x2, x3, x4 models).

![EDSR](/figs/EDSR.png)

**MDSR** (Multi-scale model. It can handle x2, x3, x4 super-resolution in a single model).

![MDSR](/figs/MDSR.png)

## NTIRE2017 Super-resolution Challenge Results

Our team (**SNU_CVLab**) won the 1st (EDSR) and 2nd (MDSR) place.

![Challenge_result](/figs/Challenge_result.png)

Following is the benchmark performance compared to previous methods.

![Paper_result](/figs/paper_result.png)


# About our code
## Dependencies
* Torch7
* cuDNN
* nccl (Optional, for faster GPU communication)

## Code
Clone this repository into any place you want. You may follow the example below.
```bash
makeReposit = [/the/directory/as/you/wish]
mkdir -p $makeReposit/; cd $makeReposit/
git clone https://github.com/LimBee/NTIRE2017.git
```

## Quick Start (Demo)
You can test the super-resolution on your own images using our trained models.

| Model | Scale | File Name | Self Esemble | Description |
| ---|---|---|---|---|
| **EDSR baseline**| x2 | baseline_x2.t7 | - | |
| **EDSR baseline**| x3 | baseline_x3.t7 | - | |
| **EDSR baseline**| x4 | baseline_x4.t7 | - | |
| **MDSR baseline**| Multi | baseline_multiscale.t7 | - | |
||||
| **EDSR**| x2 | EDSR_x2.t7 | - | |
| **EDSR**| x3 | EDSR_x3.t7 | - | |
| **EDSR**| x4 | EDSR_x4.t7 | - | |
| **MDSR**| Multi | MDSR.t7 | - | |
||||
| **EDSR+**| x2 | EDSR_x2.t7 | O | |
| **EDSR+**| x3 | EDSR_x3.t7 | O | |
| **EDSR+**| x4 | EDSR_x4.t7 | O | |
| **MDSR+**| Multi | MDSR.t7 | O | |



1. Download our models

    ```bash
    cd $makeReposit/NTIRE2017/demo/model/

    # Our models submitted to the Challenge
    wget http://cv.snu.ac.kr/research/EDSR/model_challenge.tar

    # Our models for the paper
    wget http://cv.snu.ac.kr/research/EDSR/model_paper.tar
    ```
    Or, use these links: [model_paper.tar](http://cv.snu.ac.kr/research/EDSR/model_paper.tar), 
    [model_challenge.tar](http://cv.snu.ac.kr/research/EDSR/model_paper.tar) <br>
    (**We recommend you to download the models for paper, because the models for challenge is currently not compatible with our code. Please contact us if you want to execute those models.**)

    After downloading the .tar file, make sure that the model files are placed in proper locations. For example,
    ```bash
    $makeReposit/NTIRE2017/demo/model/bicubic_x2.t7
    ```

2. Place your low-resolution test images at
    
    ```bash
    $makeReposit/NTIRE2017/demo/img_input/
    ```
    The demo code will read .jpg, .jpeg, .png format images.


3. Run `test.lua`
    
    ```bash
    cd $makeReposit/NTIRE2017/demo

    # Test EDSR (scale 2)
    th test.lua -model EDSR_x2 -selfEnsemble false

    # Test EDSR+ (scale 2)
    th test.lua -model EDSR_x2 -selfEnsemble true

    # Test MDSR (scale 2)
    th test.lua -model MDSR -scale 2 -selfEnsemble false

    # Test MDSR+ (scale 2)
    th test.lua -model MDSR -scale 2 -selfEnsemble true
    ```
    (Note: To run the **MDSR**, model name should include `multiscale`. e.g. `multiscale_blahblahblah.t7`)

    The result images will be located at
    ```bash
    $makeReposit/NTIRE2017/demo/img_output/
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

4. (Optional) Evaluate PSNR and SSIM if you have ground-truth HR images

    Place the GT images at
    ```bash
    $makeReposit/NTIRE2017/demo/img_target
    ```
    Evaluation is done by running the MATLAB script.
    ```bash
    matlab -nodisplay <evaluation.m
    ```

    If you do not want to calculate SSIM, please modify `evaluation.m` file as below. (Calculating SSIM of large image is very slow for 3 channel images.)
    ```
    line 6:     psnrOnly = false; -> psnrOnly = true;
    ```

You can reproduce our final results with `makeFinal.sh` in `NTIRE2017/demo` directory. You have to uncomment the line you want to execute.
```bash
sh makeFinal.sh
```

<!--- You can run the test script with your own model and images. Just put your images in `NTIRE2017/demo/img_input`. If you have ground-truth high-resolution images, please locate them in **NTIRE2017/demo/img_target/myData** for evaluation.

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
-->

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

* **Flickr2K** collected by ourselves using Flickr API
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

    We also provide the codes we used for downloading and selecting the Flickr2K images at
    ```bash
    $makeReposit/NTIRE2017/code/tools/Flickr2K/
    ```
    Use your own flickr API keys to use the script.

    To generate the training images for unknown track, we trained simple downsampler network.<br>
    You can download them from [here](http://cv.snu.ac.kr/research/EDSR/downsamplers.tar).

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

<br>

# Results

![result_1](/figs/result/result_1.jpg)

![result_2](/figs/result/result_2.jpg)

![result_3](/figs/result/result_3.jpg)

![result_4](/figs/result/result_4.jpg)

![result_5](/figs/result/result_5.jpg)

![result_6](/figs/result/result_6.jpg)

![result_7](/figs/result/result_7.jpg)

![result_8](/figs/result/result_8.jpg)

![result_9](/figs/result/result_9.jpg)

![result_10](/figs/result/result_10.jpg)

![result_11](/figs/result/result_11.jpg)

![result_12](/figs/result/result_12.jpg)

![result_13](/figs/result/result_13.jpg)

![result_14](/figs/result/result_14.jpg)

![result_15](/figs/result/result_15.jpg)

![result_16](/figs/result/result_16.jpg)

![result_17](/figs/result/result_17.jpg)

![result_18](/figs/result/result_18.jpg)

![result_19](/figs/result/result_19.jpg)

![result_20](/figs/result/result_20.jpg)

## Challenge: unknown downsampling track

![unknown_1](/figs/result/unknown_1.jpg)

![unknown_2](/figs/result/unknown_2.jpg)
