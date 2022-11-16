import torch
import torchaudio
from torchsummary import summary
import os 


from custom_classes import Corpus

spokencoco_dir='/home/gshen/SpokenCOCO/'
spokencoco_csv = 'spokencoco_val.csv'
spokencoco = Corpus(spokencoco_csv, root_dir = spokencoco_dir)

librispeech_root = '/home/gshen/work_dir/librispeech-train/'# 'LibriSpeech/
libri_split = 'train-clean-100' # 'test-clean'
librispeech = Corpus('librispeech_'+libri_split+'.csv', '/home/gshen/work_dir/librispeech-train/'+libri_split)

def check_weird_sents(corpus_):
    weird_sents = [x[1:] for x in corpus_ if x[2] > len(str.split(x[1]))+2]
    return weird_sents


if __name__ == "__main__":
    check_weird_sents(librispeech)