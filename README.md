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
@inproceedings{DBLP:conf/icml/LiQZ0X22,
  author    = {Jiangmeng Li and
               Wenwen Qiang and
               Changwen Zheng and
               Bing Su and
               Hui Xiong},
  editor    = {Kamalika Chaudhuri and
               Stefanie Jegelka and
               Le Song and
               Csaba Szepesv{\'{a}}ri and
               Gang Niu and
               Sivan Sabato},
  title     = {MetAug: Contrastive Learning via Meta Feature Augmentation},
  booktitle = {International Conference on Machine Learning, {ICML} 2022, 17-23 July
               2022, Baltimore, Maryland, {USA}},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  pages     = {12964--12978},
  publisher = {{PMLR}},
  year      = {2022},
  url       = {https://proceedings.mlr.press/v162/li22r.html},
  timestamp = {Tue, 12 Jul 2022 17:36:52 +0200},
  biburl    = {https://dblp.org/rec/conf/icml/LiQZ0X22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
or
```
@article{DBLP:journals/corr/abs-2203-05119,
  author    = {Jiangmeng Li and
               Wenwen Qiang and
               Changwen Zheng and
               Bing Su and
               Hui Xiong},
  title     = {MetAug: Contrastive Learning via Meta Feature Augmentation},
  journal   = {CoRR},
  volume    = {abs/2203.05119},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2203.05119},
  doi       = {10.48550/arXiv.2203.05119},
  eprinttype = {arXiv},
  eprint    = {2203.05119},
  timestamp = {Wed, 16 Mar 2022 16:41:29 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2203-05119.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
