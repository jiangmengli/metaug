# MetAug

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

We provide the checkpoint in `model_path`

## Citation

If you find this repo useful for your research, please consider citing the paper
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
