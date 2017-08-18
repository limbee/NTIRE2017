# NTIRE2017 Super-resolution Challenge: SNU_CVLab

## Introduction
This is our project repository for CVPR 2017 Workshop ([2nd NTIRE](http://www.vision.ee.ethz.ch/ntire17/)).

We, **Team SNU_CVLab**, (<i>Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah</i>, and <i>Kyoung Mu Lee</i> of [**Computer Vision Lab, Seoul National University**](http://cv.snu.ac.kr/)) are **winners** of [**NTIRE2017 Challenge on Single Image Super-Resolution**](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-CVPRW-2017.pdf). 

Our paper was published in CVPR 2017 workshop ([2nd NTIRE](http://www.vision.ee.ethz.ch/ntire17/)), and won the **Best Paper Award** of the workshop challenge track.




Please refer to our paper for details. 

If you find our work useful in your research or publication, please cite our work:

[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](http://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]
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
* Demo & Training code
* Trained models (EDSR, MDSR) 
* Datasets we used (DIV2K, Flickr2K)
* Super-resolution examples

The code is based on Facebook's Torch implementation of ResNet ([facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)). <br>

## Model Architecture
**EDSR** (Single-scale model. We provide scale x2, x3, x4 models).

![EDSR](/figs/EDSR.png)

**MDSR** (Multi-scale model. It can handle x2, x3, x4 super-resolution in a single model).

![MDSR](/figs/MDSR.png)

Note that the MDSR architecture for the challenge and for the paper[1] is slightly different.
During the challenge, MDSR had variation between two challenge tracks. While we had scale-specific feature extraction modules for track 2:unknown downscaling, we didn't use the scale-specific modules for track 1:bicubic downscaling.

**We later unified the MDSR model in our paper[1] by including scale-specific modules for both cases. From now on, unless specified as "challenge", we describe the models described in the paper.**

## NTIRE2017 Super-resolution Challenge Results

We proposed 2 methods and they won the 1st (EDSR) and 2nd (MDSR) place.

![Challenge_result](/figs/Challenge_result.png)

We have also compared the super-resolution performance of our models with previous state-of-the-art methods.

![Paper_result](/figs/paper_result.png)

# About our code
## Dependencies
* Torch7
* cuDNN
* nccl (Optional, for faster GPU communication)

Our code is tested under Ubuntu 14.04 and 16.04 environment with Titan X GPUs (12GB VRAM).

## Code
Clone this repository into any place you want. You may follow the example below.
```bash
makeReposit = [/the/directory/as/you/wish]
mkdir -p $makeReposit/; cd $makeReposit/
git clone https://github.com/LimBee/NTIRE2017.git
```

## Quick Start (Demo)
You can test our super-resolution algorithm with your own images.

We assume the images are downsampled by bicubic interpolation.

| Model | Scale | File Name | Self Esemble | # ResBlocks | # Filters | # Parameters |
|  ---  |  ---  | ---       | ---          | ---         |---        |---           |
| **EDSR baseline**| x2 | baseline_x2.t7 | X | 16 | 64 | 1.5M |
| **EDSR baseline**| x3 | baseline_x3.t7 | X | 16 | 64 | 1.5M | 
| **EDSR baseline**| x4 | baseline_x4.t7 | X | 16 | 64 | 1.5M | 
| **MDSR baseline**| Multi | baseline_multiscale.t7 | X | 16 | 64 | 3.2M |
||||||||
| **EDSR**| x2 | EDSR_x2.t7 | X | 32 | 256 | 43M | 
| **EDSR**| x3 | EDSR_x3.t7 | X | 32 | 256 | 43M | 
| **EDSR**| x4 | EDSR_x4.t7 | X | 32 | 256 | 43M | 
| **MDSR**| Multi | MDSR.t7 | X | 80 | 64 | 8.0M |
||||||||
| **EDSR+**| x2 | EDSR_x2.t7 | O | 32 | 256 | 43M | 
| **EDSR+**| x3 | EDSR_x3.t7 | O | 32 | 256 | 43M | 
| **EDSR+**| x4 | EDSR_x4.t7 | O | 32 | 256 | 43M | 
| **MDSR+**| Multi | MDSR.t7 | O | 80 | 64 | 8.0M |

<br>

1. Download our models

    ```bash
    cd $makeReposit/NTIRE2017/demo/model/

    # Our models for the paper[1]
    wget http://cv.snu.ac.kr/research/EDSR/model_paper.tar
    ```

    Or, use the link: [model_paper.tar](http://cv.snu.ac.kr/research/EDSR/model_paper.tar)
    <!-- [model_challenge.tar](http://cv.snu.ac.kr/research/EDSR/model_paper.tar) <br> -->
    (**If you would like to run the models we used during the challenge, please contact us.**)

    After downloading the .tar files, make sure that the model files are placed in proper locations. For example,

    ```bash
    $makeReposit/NTIRE2017/demo/model/bicubic_x2.t7
    $makeReposit/NTIRE2017/demo/model/bicubic_x3.t7
    ...
    ```

2. Place your low-resolution test images at
    
    ```bash
    $makeReposit/NTIRE2017/demo/img_input/
    ```
    The demo code will read .jpg, .jpeg, .png format images.


3. Run `test.lua`
    
    **You can run different models and scales by changing input arguments.**
    
    ```bash
    # To run for scale 2, 3, or 4, set -scale as 2, 3, or 4
    # To run EDSR+ and MDSR+, you need to set -selfEnsemble as true

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
    (Note: To run the **MDSR**, model name should include `multiscale` or `MDSR`. e.g. `multiscale_blahblahblah.t7`)

    The result images will be located at
    ```bash
    $makeReposit/NTIRE2017/demo/img_output/
    ```

    * Here are some optional argument examples you can adjust. Please refer to the following explanation.

    ```bash
    # You can test our model with multiple GPU. (n = 1, 2, 4)
    -nGPU       [n]

    # You must specify this directory. Default is /var/tmp/dataset
    -dataDir    [$makeData]
    -dataset    [DIV2K | myData]
    -save       [Folder name]

    # Please see our paper[1] if you want to know about self-ensemble.
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

    If you don't want to calculate SSIM, please modify `evaluation.m` file as below. (Calculating SSIM of large image is very slow for 3 channel images.)
    ```
    line 6:     psnrOnly = false; -> psnrOnly = true;
    ```

You can reproduce our final results by running `makeFinal.sh` in `NTIRE2017/demo` directory. Please uncomment the command you want to execute in the file.
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
If you want to train or evaluate our models with DIV2K or Flickr2K dataset, please download the dataset from [here](http://cv.snu.ac.kr/research/EDSR/DIV2K.tar).
Place the tar file to the location you want. **(We recommend /var/tmp/dataset/)**  <U>If the dataset is located otherwise, **you have to change the optional argument -dataset for training and test.**</U>

* [**DIV2K**](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf) from [**NTIRE2017**](http://www.vision.ee.ethz.ch/ntire17/)
    ```bash
    makeData = /var/tmp/dataset/ # We recommend this path, but you can freely change it.
    mkdir -p $makeData/; cd $makedata/
    tar -xvf DIV2K.tar
    ```
    You should have the following directory structure:

    `/var/tmp/dataset/DIV2K/DIV2K_train_HR/0???.png`<br>
    `/var/tmp/dataset/DIV2K/DIV2K_train_LR_bicubic/X?/0???.png`<br>
    `/var/tmp/dataset/DIV2K/DIV2K_train_LR_unknown/X?/0???.png`<br>

* **Flickr2K** dataset collected by ourselves using Flickr API
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

    We also provide the code we used for collecting the Flickr2K images at
    ```bash
    $makeReposit/NTIRE2017/code/tools/Flickr2K/
    ```
    Use your own flickr API keys to use the script.

    During the challenge, we additionally generated training data by learning simple downsampler networks from DIV2K dataset track 2.<br>
    You can download the downsampler models from [here](http://cv.snu.ac.kr/research/EDSR/downsamplers.tar).

To make data loading faster, you can convert the dataset into binary .t7 files
* Convert **DIV2K** dataset from .png to into .t7 files
    ```bash
    cd $makeReposit/NTIRE2017/code/tools

    # Choose one among below

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
You can also use .png files too. Please see below **Training** section for the details.

## Training

1. To train our baseline model, please run the following command:

    ```bash
    th main.lua         # This model is not our final model!
    ```

    * Here are some optional arguments you can adjust. If you have any problem, please refer following lines. You can check out details in `NTIRE2017/code/opts.lua`.
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

    <U>Some model may require pre-trained **bicubic scale 2** or **bicubic multiscale** model.</U> Here, we assume that you already downloaded `bicubic_x2.t7` and `bicubic_multiscale.t7` in the `NTIRE2017/demo/model` directory. Otherwise, you can create them yourself. It is also possible to start the traning from scratch by removing `-preTrained` option in `training.sh`.

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

## NTIRE2017 SR Challenge: Unknown Down-sampling Track

![unknown_1](/figs/result/unknown_1.jpg)

![unknown_2](/figs/result/unknown_2.jpg)
