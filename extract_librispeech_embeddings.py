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


import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
librispeech_root = 'librispeech-train/'# 'LibriSpeech/
libri_split = 'train-clean-100' # 'test-clean'
saved_file = libri_split+'-extracted.pt'
sr = 16000


model_file = 'wav2vec_small.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torchaudio.models.wav2vec2.utils import import_fairseq_model
model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
original = model[0]
imported = import_fairseq_model(original)

from custom_classes import LibriDataset
from custom_functions import walk_librispeech_dirs


def generating_features(dataset):
    feat_list = []
    lab_list = []
    for waveform, sent in dataset:
        doc = nlp(sent)
        for sent in doc.sentences:
            depth = sent.constituency.depth()
            if depth < 20:
                lab_list.append(depth)
                with torch.inference_mode():
                    features, _ = imported.to(device).extract_features(waveform.to(device))
                    # print(features)
                    audio_len = len(waveform[-1])/sr
                    features = [torch.mean(x.cpu(),dim=1).squeeze().numpy() for x in features]
                    features = [np.append(layer,[audio_len]) for layer in features]
                    # feat_list.append(torch.mean(features,dim=1).squeeze().numpy())
                    feat_list.append(features)

    return feat_list, lab_list

# def making_libri_csv(librispeech_root,libri_split):
    libri_split_path = os.path.join(librispeech_root,libri_split)
    libri_sub1 = []
    for i in os.listdir(libri_split_path):
        libri_sub1.append(os.path.join(libri_split_path,i))

    libri_sub2 = []
    for i in libri_sub1:
        # print(i)
        for j in os.listdir(i):
            libri_sub2.append(os.path.join(i,j))

    audio_list = []
    for sub_dir in libri_sub2:
        temp_file_list = os.listdir(sub_dir)
        temp_audio_list = [os.path.join(sub_dir,x) for x in temp_file_list if 'flac' in x]
        audio_list += temp_audio_list

    df = pd.DataFrame(audio_list)
    df['dirname'] = df[0].apply(lambda x : os.path.split(x)[0])
    df['fileid'] = df[0].apply(lambda x : os.path.split(x)[1])
    df['fileid'] = df['fileid'].str.replace('.flac', '')
    df = df.rename({0:'fullpath'},axis=1)

    filepath = libri_split_path
    path_list = []
    for subdir1 in os.listdir(filepath):
        # print(f"there are {len(subdir1)} different speakers")
        temp_dir_list = os.listdir(os.path.join(filepath,subdir1))
        for subdir2 in temp_dir_list:
            # print(f"there are {len(subdir2)} different chapters for speaker {subdir1}")
            path_list.append(os.path.join(os.path.join(filepath,subdir1),subdir2))
    txtfiles = []
    for txt_path in path_list:
        txtfiles.append([os.path.join(txt_path, x) for x in os.listdir(txt_path) if 'txt' in x][0])

    df_txt = pd.concat([pd.read_csv(item, header=None) for item in txtfiles],ignore_index=True)
    df_txt['fileid'] = df_txt[0].str.extract('(\d+-\d+-\d+)')
    df_txt['sent'] = df_txt[0].str.replace(r'(\d+-\d+-\d+\s)', '')
    df_txt = df_txt.drop([0], axis=1)

    merge_df = pd.merge(df,df_txt, on = 'fileid')
    merge_df.to_csv('librispeech_'+libri_split+'.csv', index=False)


if __name__ == "__main__":
    df = walk_librispeech_dirs(librispeech_root=librispeech_root, libri_split=libri_split)
    df.to_csv('librispeech_'+libri_split+'.csv', index=False)
    print(f"generating features")
    libri_ds = LibriDataset('librispeech_'+libri_split+'.csv', '/home/gshen/work_dir/')
    embeddings, labels = generating_features(libri_ds)
    print(f'zipping stuff together')
    extracted = list(zip(embeddings, labels))
    print(f"saving the extracted embeddings to {saved_file}")
    torch.save(extracted, saved_file)

