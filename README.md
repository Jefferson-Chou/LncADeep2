# Introduction
LncADeep 2.0, an advanced tool that identifies lncRNA transcripts based on deep learning and leverages transfer learning to functionally annotate lncRNAs. 

# Build LncADeep 2.0
```
cd ~/
git clone https://github.com/Jefferson-Chou/LncADeep2.git
```
## Create a new conda environment
Install [Anaconda](https://www.anaconda.com/download/success) then:
```
conda create --name lncadeep python=3.9
conda activate lncadeep
```

## Install the necessary dependencies
### Install PyTorch and PyG
Check the CUDA version if GPU is available on your device
```
nvidia-smi
```
then install the compatible version of [latest version](https://pytorch.org/get-started/locally/) or [previous versions](https://pytorch.org/get-started/previous-versions/) of PyTorch and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
```
# The test platform is Ubuntu 18.04 with CUDA version 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0%2Bcu117.html
pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu117/pyg_lib-0.3.1%2Bpt113cu117-cp39-cp39-linux_x86_64.whl 
```
### Install other dependencies and R packages
```
# In Linux terminal:
cd ~/LncADeep2/
pip install -r ./requirements.txt
conda install -c conda-forge r-base==4.3.3
```
R packages used are listed in `R packages versions.csv`
### Note
Files mentioned below can be obtained in [Zenodo](https://doi.org/10.5281/zenodo.17667433).
* The case study section is based on the R packages listed in `Rpkg.tar.gz`. Users can install these packages simply by:
```
tar xvf Rpkg.tar.gz
cp -r ./Rpkg/* ~/.conda/envs/lncadeep/lib/R/library/
```
* The lncRNA function annotation module calls for the DNA-BERT pre-trained model, which can be obtained in [huggingface](https://huggingface.co/zhihan1996/DNABERT-2-117M). Users can also choose to download the file named  `DNABERT-2-117M.tar.gz` in [Zenodo](https://zenodo.org/records/14882729?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJjNWM5NDZhLTZjZTYtNGZjZi1hNzBmLWNmNWVlOGQxYzMyYiIsImRhdGEiOnt9LCJyYW5kb20iOiIzYjZkZjk3MzE0OTI2ZmFlYTA5ZmI4MDdhY2JiYTlhYSJ9.oekKwDbkjTxlzI1SRs5g0bPbNRNZiAheGliFvrWuZEV0n6sgX6fhzRhTgebJeGlf8LPl1WUdfRaxKJ_AVnzGtw). After downloaded, please:
```
cd ~/LncADeep2/annotation/models/
tar xvf DNABERT-2-117M.tar.gz
```
* The lncRNA identification module calls for Pfam dataset and hmmsearch, which are named `identification_src.tar.gz` in [Zenodo](https://zenodo.org/records/14882729?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJjNWM5NDZhLTZjZTYtNGZjZi1hNzBmLWNmNWVlOGQxYzMyYiIsImRhdGEiOnt9LCJyYW5kb20iOiIzYjZkZjk3MzE0OTI2ZmFlYTA5ZmI4MDdhY2JiYTlhYSJ9.oekKwDbkjTxlzI1SRs5g0bPbNRNZiAheGliFvrWuZEV0n6sgX6fhzRhTgebJeGlf8LPl1WUdfRaxKJ_AVnzGtw). After downloaded, please:
```
cd ~/LncADeep2/identification
tar xvf identification_src.tar.gz
```
* The lncRNA annotation module dependencies `annotation_src.tar.gz` can be downloaded in [Zenodo](https://zenodo.org/records/14882729?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJjNWM5NDZhLTZjZTYtNGZjZi1hNzBmLWNmNWVlOGQxYzMyYiIsImRhdGEiOnt9LCJyYW5kb20iOiIzYjZkZjk3MzE0OTI2ZmFlYTA5ZmI4MDdhY2JiYTlhYSJ9.oekKwDbkjTxlzI1SRs5g0bPbNRNZiAheGliFvrWuZEV0n6sgX6fhzRhTgebJeGlf8LPl1WUdfRaxKJ_AVnzGtw). After downloaded, please:
```
cd ~/LncADeep2/annotation
tar xvf annotation_src.tar.gz
```
# Quick start
```
cd ~/LncADeep2
python LncADeep2.py -h
```
## Options
* (__required__): `--mode` specifies the mode of LncADeep2, `identify` mode to distinguish lncRNA transcripts from mRNA ones, `anno` mode to perform functional annotations for input lncRNAs sequences, `train` mode to train a new identification model. 
  
* (__required__): `--input` is the path to the input fasta file.
  
* (__required__): `--output` is the path to the output file, default stored in /identification/output/ in the `identify` mode and /annotation/output/ in the `anno` mode.

* (__optional__): `--thread` refers to the number of threads of hmmer in `identify` mode and GO enrichment in `anno` mode, default: 1.

* (__optional__): `--device` is the GPU or CPU device used for identification or annotation, default: 'cpu'.

* (__optional__): `--feature` is a binary variable(`0` for no and `1` for yes) to determine whether to save features files (can be used for transfer learning for instance) in /identification/output/, default: 0.

* (__optional__): `--model_file` specifies the path to the model file (for example, the model trained by `train` mode) for identification, default: /identification/models/model_93_0.1750_0.9440.pt.
  
* (__optional__): `--training_labels` is the txt file with labels corresponding to the training FASTA file in the `train` mode.

## Input requirements and recommendations
* In the `identify` mode, the input should be a FASTA file.
* In the `train` mode, the inputs should include a FASTA file with corresponding label file.
* In the `anno` mode, the inputs should be a FASTA file whose headers started with '>' are highly recommended to be assigned as ensembl IDs (e.g. ENSG00000223573)

## Examples
```
cd ~/LncADeep2/
# 'identify' mode
python LncADeep2.py -m identify -i ./identification/test_data/lncRNA_mRNA_test.fa -th 8 -o ./identification/output/test.csv -d 'cuda:0' -fe 1

# 'train' mode
python LncADeep2.py -m train -i ./identification/test_data/lncRNA_mRNA_test.fa -tl ./identification/test_data/lncRNA_mRNA_test.txt -d 'cuda:0' -th 8

# 'anno' mode
python LncADeep2.py -m anno -i ./annotation/test_data/test2.fa -d 'cuda:1' -th 10 -o ./annotation/output/
```
# Contact
If you have any questions, please ask us: pkuzhou@stu.pku.edu.cn or hqzhu@pku.edu.cn
