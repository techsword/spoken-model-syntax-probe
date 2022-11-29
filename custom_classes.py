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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Corpus(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the Librispeech directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audiofilelist = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.audiofilelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root_dir,
                                self.audiofilelist.iloc[idx, 0])
        audio, sr = torchaudio.load(audio_name)
        annot = self.audiofilelist.iloc[idx,-1]
        # sample = {'audio': audio.to(device), 'file': audio_name, 'sr': sr, 'annot': sent}
        sample = audio.to(device)

        doc = nlp(annot)
        for sent in doc.sentences:
            depth = sent.constituency.depth()
        

        if self.transform:
            sample = self.transform(sample)

        return sample, annot, depth, audio_name

    def collate(self, batch):
        return batch
    
    def get_depth(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        annot = self.audiofilelist.iloc[idx,-1]
        doc = nlp(annot)
        for sent in doc.sentences:
            depth = sent.constituency.depth()
        return annot, depth
