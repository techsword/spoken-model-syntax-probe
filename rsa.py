import random

# nltk.download('punkt')
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import os

from scipy.stats import pearsonr


kernel_lookup = {'scc_tree_kernel': '/home/gshen/work_dir/spoken-model-syntax-probe/scc_trees_kernel.pt',
'libri_tree_kernel': '/home/gshen/work_dir/spoken-model-syntax-probe/libri_trees_kernel.pt'}

pairwise_distance_path = '/home/gshen/work_dir/spoken-model-syntax-probe/pairwise_distances'

pairwise_distance_files = [os.path.join(pairwise_distance_path,x) for x in os.listdir(pairwise_distance_path)]

if __name__ == "__main__":
    for x in pairwise_distance_files:
        # print(x)
        if 'librispeech' in x:
            kernel = [x[0] for x in torch.load(kernel_lookup['libri_tree_kernel'])]

        elif 'spokencoco' in x:
            kernel = [x[0] for x in torch.load(kernel_lookup['scc_tree_kernel'])]
            
        calculated_distances = torch.load(x)
            
        for layer in calculated_distances:
            r_score = pearsonr(kernel, calculated_distances[layer])
            print((os.path.basename(x)[:-30], layer) + r_score)