import nltk
import torch
from sklearn.preprocessing import normalize

nltk.download('punkt')
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import time
from itertools import combinations, combinations_with_replacement, product

import pandas as pd
from nltk.tree import Tree
from sklearn.metrics import pairwise_distances

from ursa.kernel import Kernel

tree_paths = {  "scc_trees_path": '/home/gshen/work_dir/spoken-model-syntax-probe/scc_generated_trees.pt', 
                "libri_trees_path": '/home/gshen/work_dir/spoken-model-syntax-probe/libri_generated_trees.pt'}

def calculate_kernel(tree_list, normalization = False):    
    K = Kernel()
    tree_kernel = []
    for i in combinations_with_replacement([x[0] for x in tree_list], 2):
        kernel = K(i[0], i[1])
        if normalization == True:
            denom = (K(i[0], i[0])*K(i[1],i[1]))**0.5
        else:
            denom = 1
        tree_kernel.append(kernel/denom)
        # print(K(i[0], i[1]))
        # print(i[0],i[1])
    num_entries = len(tree_list)
    # tree_kernel = np.reshape(tree_kernel,(num_entries,num_entries)).T

    df = pd.DataFrame(tree_kernel)
    return df


if __name__ == "__main__":
    
    select_ = 'libri_trees_path'

    tree_list = torch.load(tree_paths[select_])
    st = time.time()
    df = calculate_kernel(tree_list)
    df.to_csv(select_[:-4]+'kernel.csv')
    et = time.time()
    print('script ran for', et-st, 'seconds')