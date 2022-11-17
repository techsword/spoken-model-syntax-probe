# spoken-model-syntax-probe

This probing project aims to probe the syntax encoded in several spoken language models.

## Installation

Clone repo and set up and activate a virtual environment with python3

```
cd spoken-model-syntax-probe
virtualenv -p python3 .
```

Install pre-requisites annd the Python code.

```
pip install -r requirements.txt
```

Download HuBERT and wav2vec 2.0 moodels and data and unpack them:

```
wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt

and/or

wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
```


## Running TreeDepth probe

### Extract embeddings from spoken language model

Both wav2vec 2.0 and HuBERT extraction have been implemented in `extract_embeddings.py` 

It is possible to use `python extract_embeddings.py -h` to see what parameters can be passed through the python script. The script will save the extracted features including the flattened embedding, the treedepth, the annotation, and the audio path to a .pt file.

You may need to modify the paths that point to the downloaded model files in `extract_embeddings.py` under `model_dict`

#### Datasets

This project have implemented the embedding extraction script for LibriSpeech and SpokenCOCO corpus. You can download the two corpora from the links here [SpokenCOCO](https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz) [LibriSpeech](https://www.openslr.org/12). 

For LibriSpeech: The extraction script is designed to work with different splits of the LibriSpeech corpus. You can manually define the root directory (`librispeech_root`) and the split (`libri_split`) in `extract_embeddings.py`. The script will read all the sub-directories and generate a csv file that can be read by the dataset class.

For SpokenCOCO: The extraction script takes the json file that comes with the SpokenCOCO corpus and generates a csv file that can be read by the dataset class. You can define the json path (`json_path`) in `extract_embeddings.py` manually.



### Running sklearn models to predict TreeDepth

After obtaining the embeddings, you may want to check if the `dataset_dict` in `model_run.py` is accurate before running the models. You can read the parser arguments to find out more about the structures of the script.

##### Usage in SLURM environment

Use `sbatch --array=[index value] slurm.sh` to submit a job array to SLURM environments for training the regression models.

ref: https://slurm.schedmd.com/job_array.html

