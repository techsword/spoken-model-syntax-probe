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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


sr = 16000

import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

def loading_fairseq_model(model_file):
    from torchaudio.models.wav2vec2.utils import import_fairseq_model
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
    original = model[0]
    imported = import_fairseq_model(original)
    return imported

def generating_features(dataset, model_file):
    feat_list = []
    lab_list = []
    annot_list = []
    wav_list = []
    model = loading_fairseq_model(model_file=model_file)
    for waveform, annot in dataset:
        doc = nlp(annot)
        for sent in doc.sentences:
            depth = sent.constituency.depth()
            wordcount = len(sent.words)
            if depth < 13 and depth > 5:
                lab_list.append(depth)
                annot_list.append(annot)
                wav_list.append(waveform)
                with torch.inference_mode():
                    features, _ = model.to(device).extract_features(waveform.to(device))
                    # print(features)
                    audio_len = len(waveform[-1])/sr
                    features = [torch.mean(x.cpu(),dim=1).squeeze().numpy() for x in features]
                    features = [np.append(layer,[audio_len]) for layer in features]
                    features = [np.append(layer,[wordcount]) for layer in features]
                    # feat_list.append(torch.mean(features,dim=1).squeeze().numpy())
                    feat_list.append(features)

    
    print(f"there are {len(lab_list)} in the extracted dataset, each tensor is {features[0].shape}, the max tree depth is {max(lab_list)} and the min is {min(lab_list)}")
    return list(zip(feat_list, lab_list,annot_list,wav_list))

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

