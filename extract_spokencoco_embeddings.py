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

import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

model_file = 'wav2vec_small.pt'
sr = 16000
json_path = '/home/gshen/SpokenCOCO/SpokenCOCO_val.json'
root_dir='/home/gshen/SpokenCOCO/'
saved_file = 'spokencoco_extracted.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
csv_file = 'spokencoco_val.csv'

# importing fairseq pretrained modelfile to pytorch
from torchaudio.models.wav2vec2.utils import import_fairseq_model
model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
original = model[0]
imported = import_fairseq_model(original)


def read_json_save_csv(json_path, csv_file):
    '''Reading the SpokenCOCO json file and extracting the text and wav file pairs'''

    with open(json_path) as json_file:
        data = json.load(json_file)
    text = []
    wav = []
    for image in data['data']:
        for caption in image['captions']:
            text.append(caption['text'])
            wav.append(caption['wav'])


    '''
    Using pandas to combine the two lists extracted in the cells above and 
    save the resulting dataframe into a csv file
    '''


    dict_spokencoco = dict(zip(text,wav))
    spokencoco_val = pd.Series(dict_spokencoco, name = 'wav')
    spokencoco_val.index.name = 'text'
    spokencoco_val = spokencoco_val.reset_index()
    column_titles = ['wav', 'text']
    spokencoco_val = spokencoco_val.reindex(columns = column_titles)
    spokencoco_val.to_csv(csv_file, header=None, index = None)

spokencoco = SpokenCOCODataset(csv_file, root_dir = root_dir)

def generating_features(dataset):
    feat_list = []
    lab_list = []
    for waveform, sent in dataset:
        doc = nlp(sent)
        for sent in doc.sentences:
            depth = sent.constituency.depth()
            wordcount = len(sent.words)
            if depth < 20:
                lab_list.append(depth)
                with torch.inference_mode():
                    features, _ = imported.to(device).extract_features(waveform.to(device))
                    # print(features)
                    audio_len = len(waveform[-1])/sr
                    features = [torch.mean(x.cpu(),dim=1).squeeze().numpy() for x in features]
                    features = [np.append(layer,[audio_len]) for layer in features]
                    features = [np.append(layer,[wordcount]) for layer in features]
                    # feat_list.append(torch.mean(features,dim=1).squeeze().numpy())
                    feat_list.append(features)

    return feat_list, lab_list

if __name__ == "__main__":
    print(f"generating features")
    embeddings, labels = generating_features(spokencoco)
    print(f'zipping stuff together')
    spokencoco_extracted = list(zip(embeddings, labels))
    print(f"saving the extracted embeddings to {saved_file}")
    torch.save(spokencoco_extracted, saved_file)
