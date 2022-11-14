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

from custom_classes import SpokenCOCODataset

# import stanza
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

model_file = 'hubert_base_ls960.pt' # 'hubert_base_ls960.pt' or 'wav2vec_small.pt'
sr = 16000
json_path = '/home/gshen/SpokenCOCO/SpokenCOCO_val.json'
root_dir='/home/gshen/SpokenCOCO/'
saved_file = 'spokencoco_extracted_' + model_file[:7] + '.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
csv_file = 'spokencoco_val.csv'

# # importing fairseq pretrained modelfile to pytorch
# from torchaudio.models.wav2vec2.utils import import_fairseq_model
# model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
# original = model[0]
# imported = import_fairseq_model(original)


from custom_functions import read_json_save_csv, generating_features



if __name__ == "__main__":
    read_json_save_csv(json_path,csv_file)
    print(f"generating features")
    spokencoco = SpokenCOCODataset(csv_file, root_dir = root_dir)
    spokencoco_extracted = generating_features(spokencoco, model_file)
    print(f"saving the extracted embeddings to {saved_file}")
    torch.save(spokencoco_extracted, saved_file)
