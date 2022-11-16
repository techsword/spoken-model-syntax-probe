import torch
import torchaudio
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import fairseq
import numpy as np
import pandas as pd
import os 
import pickle
import re
import json


from custom_classes import Corpus
from custom_functions import walk_librispeech_dirs, read_json_save_csv, generating_features

model_file = '/home/gshen/work_dir/spoken-model-syntax-probe/hubert_base_ls960.pt' #'/home/gshen/work_dir/wav2vec_small.pt' # 'hubert_base_ls960.pt' or 'wav2vec_small.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sr = 16000


# librispeech settings
librispeech_root = '/home/gshen/work_dir/librispeech-train/'# 'LibriSpeech/
libri_split = 'train-clean-100' # 'test-clean'
saved_file = libri_split+'-extracted.pt'



# spokencoco settings

json_path = '/home/gshen/SpokenCOCO/SpokenCOCO_val.json'
root_dir='/home/gshen/SpokenCOCO/'
saved_file = os.path.basename(model_file[:-3]) + '_spokencoco_extracted' + '.pt'
csv_file = 'spokencoco_val.csv'



if __name__ == "__main__":
    if os.path.isfile(csv_file) == False:
        print(f"{csv_file} not found, creating from {json_path}")
        spokencoco_df = read_json_save_csv(json_path)
        spokencoco_df.to_csv(csv_file, header=None, index = None)
    elif os.path.isfile('librispeech_'+libri_split+'.csv') == False:
        librispeech_dataset_df = walk_librispeech_dirs(librispeech_root=librispeech_root, libri_split=libri_split)
        librispeech_dataset_df.to_csv('librispeech_'+libri_split+'.csv', index=False)

    print(f"generating features")
    libri_ds = Corpus('librispeech_'+libri_split+'.csv', '/home/gshen/work_dir/')
    libri_extracted = generating_features(libri_ds)
    print(f'zipping stuff together')
    print(f"saving the extracted embeddings to {saved_file}")
    torch.save(libri_extracted, saved_file)


    print(f"generating features")
    spokencoco = Corpus(csv_file, root_dir = root_dir)
    spokencoco_extracted = generating_features(spokencoco, model_file)
    print(f"saving the extracted embeddings to {saved_file}")
    torch.save(spokencoco_extracted, saved_file)
