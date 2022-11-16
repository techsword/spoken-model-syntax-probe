import torch
import torchaudio
from torchsummary import summary
import os 


from custom_classes import SpokenCOCODataset

root_dir='/home/gshen/SpokenCOCO/'
csv_file = 'spokencoco_val.csv'
work_dir = '/home/gshen/work_dir/'
spokencoco = SpokenCOCODataset(os.path.join(work_dir,csv_file), root_dir = root_dir)
weird_sents = [x[1:] for x in spokencoco if x[2] > len(str.split(x[1]))+2]

print(weird_sents)