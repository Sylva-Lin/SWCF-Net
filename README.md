SWCF-Net is accepted by IROS'24 🎉🎉🎉

<div align="center">
<h1>SWCF-Net</h1>
<h3>Similarity-weighted Convolution and Local-global Fusion for Efficient Large-scale Point Cloud Semantic Segmentation</h3>

# Overview
</div>

<div  align="center">    
 <img src="./figure/Arc.png" width = ""  align=center />
</div>

## 1. Environment
The code has been tested on Ubuntu 20.04 with 1 Nvidia 3090 GPUs (24GB memory).

1. Python 3.8.16
    ```bash
    conda create -n your_env_name python=3.8.16
    ```

2. Install torch 1.10.0 + cu113

    ```bash
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113
    ```

3. Clone this repository and install the requirements.

    ```bash
    pip install -r requirements.txt
    ```

## 2. SemanticKITTI Segmentation

1. **Data**: SemanticKITTI dataset can be found [here](http://semantic-kitti.org/dataset.html#download). Download the files related to semantic segmentation and extract everything into the same folder. Uncompress the folder and move it to ```/data/semantic_kitti/dataset```.
   Run the following command to prepare the dataset.

    ```bash
    python utils/data_prepare_semantickitti.py
    ```
    
2. **Train**: 

    ```bash
    python train_SemanticKITTI.py <args>
    ```

3. **Test**: 

    ```bash
    python test_SemanticKITTI.py <args>
    ```

## 3. Result
