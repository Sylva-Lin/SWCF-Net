# SWCF-Net
SWCF-Net is accepted by IROS 2024 🎉🎉🎉

## Environment
The code has been tested on Ubuntu 20.04 with 1 Nvidia 3090 GPU (24GB memory).

1. Python 3.8.16
    ```bash
    conda create -n your_env_name python=3.8.16 -y
    ```

2. Install torch 1.10.0 + cu113
    ```bash
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113
    ```

3. nearest_neighbors && cpp_wrappers
   ```bash
   cd utils/nearest_neighbors
   python setup.py install --home="."
   cd ../../
   ```
   ```bash
   cd utils/cpp_wrappers
   sh compile_wrappers.sh
   cd ../../../
   ```

## SemanticKITTI Segmentation

1. **Data**: SemanticKITTI dataset can be found [here](http://semantic-kitti.org/dataset.html#download). Download the files related to semantic segmentation and extract everything into the same folder. Uncompress the folder and move it to ```/data/semantic_kitti/dataset```.
   Run the following command to prepare the dataset.

    ```bash
    python data_prepare.py --src_path your_dataset_path --dst_path your_process_path
    ```
    
2. **Train**: 

    ```bash
    python train_SemanticKITTI.py --log_dir your_result_path
    ```

## Citation
If you find _SWCF-Net_ useful to your research, please cite our work as an acknowledgment.
```bib
@article{lin2024swcf,
  title={SWCF-Net: Similarity-weighted Convolution and Local-global Fusion for Efficient Large-scale Point Cloud Semantic Segmentation},
  author={Lin, Zhenchao and He, Li and Yang, Hongqiang and Sun, Xiaoqun and Zhang, Cuojin and Chen, Weinan and Guan, Yisheng and Zhang, Hong},
  journal={arXiv preprint arXiv:2406.11441},
  year={2024}
}
```
