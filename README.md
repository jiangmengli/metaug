# MetAug

This paper is accepted by ICML2022.

### Install the PyTorch based environment for MetAug. 
```bash
# Create a conda environment
conda create -n metaug python=3.7

# Activate the environment
conda activate metaug

# Install dependencies
pip install -r requirements.txt
```

### Install the datasets.
The used datasets are totally established, please follow the official instruction to install the datasets. Note that our code requires the datasets containing the "images" not "features".

## How to Run

We provide the running scripts as follows. Make sure you change the paths in `data_folder`, `model_path`, and `tb_path` and run the commands.
```bash
# Train
python train_MetAug.py

# Test
python LinearProbing_MetAug.py
```

Readers can change hyperparameters directly in code or bash script

## Checkpoint

We provide the checkpoint in `https://drive.google.com/file/d/1uClfQ3u_3U3Kag-a0SSs2LRRI3qW4OA6/view?usp=sharing`

## Citation

If you find this repo useful for your research, please consider citing the paper
```
@article{li2022metaug,
  title={MetAug: Contrastive Learning via Meta Feature Augmentation},
  author={Li, Jiangmeng and Qiang, Wenwen and Zheng, Changwen and Su, Bing and Xiong, Hui},
  journal={arXiv preprint arXiv:2203.05119},
  year={2022}
}
```
or
```
@inproceedings{jml2022metaug,
  author    = {Jiangmeng Li and
               Wenwen Qiang and
               Changwen Zheng and
               Bing Su and
               Hui Xiong},
  title     = {MetAug: Contrastive Learning via Meta Feature Augmentation},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning,
               {ICML} 2022, 17-23 July 2022, Baltimore, Maryland, {USA}},
  series    = {Proceedings of Machine Learning Research},
  publisher = {{PMLR}},
  year      = {2022},
}
```
