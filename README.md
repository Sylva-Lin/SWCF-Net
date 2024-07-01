# SWCF-Net
SWCF-Net is accepted by IROS 2024 🎉🎉🎉

## 1. Environment
The code has been tested on Ubuntu 20.04 with 1 Nvidia 3090 GPU (24GB memory).

1. Python 3.8.16
    ```bash
    conda create -n your_env_name python=3.8.16 -y
    ```

2. Install torch 1.10.0 + cu113

    ```bash
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113
    ```

## 2. SemanticKITTI Segmentation

1. **Data**: SemanticKITTI dataset can be found [here](http://semantic-kitti.org/dataset.html#download). Download the files related to semantic segmentation and extract everything into the same folder. Uncompress the folder and move it to ```/data/semantic_kitti/dataset```.
   Run the following command to prepare the dataset.

    ```bash
    python data_prepare.py --src_path your_dataset_path --dst_path your_process_path
    ```
    
2. **Train**: 

    ```bash
    python train_SemanticKITTI.py --log_dir your_result_path
    ```
