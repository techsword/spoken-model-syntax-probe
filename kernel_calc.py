import nltk
import torch
import random
random.seed(42)

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

# def calculate_kernel(tree_list, normalization = False):    
#     K = Kernel()
#     tree_kernel = []
#     # random.shuffle(tree_list)
#     num_sents = len(tree_list)
#     # if num_sents % 2 != 0:
#         # num_sents -= 1
#     # sublist_1, sublist_2 = tree_list[:num_sents/2], tree_list[num_sents/2:]
#     # for i in combinations_with_replacement([x[0] for x in tree_list], 2):
#     for i in 
#         kernel = K(i[0], i[1])
#         if normalization == True:
#             denom = (K(i[0], i[0])*K(i[1],i[1]))**0.5
#         else:
#             denom = 1
#         tree_kernel.append(kernel/denom)
#         # print(K(i[0], i[1]))
#         # print(i[0],i[1])
#     num_entries = len(tree_list)
#     # tree_kernel = np.reshape(tree_kernel,(num_entries,num_entries)).T

#     df = pd.DataFrame(tree_kernel)
#     return df

def calculate_kernel(tree_list, normalization = True):
    K = Kernel()
    tree_kernel = []
    for i in range(0, len(tree_list), 2):
        tree_1, tree_2 = tree_list[i][0], tree_list[i+1][0]
        kern = K(tree_1, tree_2)
        if normalization == True:
            # Normalize the kernels calculated 
            denom = (K(tree_1, tree_1)*K(tree_2,tree_2))**0.5
        else:
            denom = 1

        normed_kern = kern/denom

        tree_kernel.append((normed_kern,tree_list[i][0],tree_list[i+1][0]))
        # print(i)
    return tree_kernel


if __name__ == "__main__":
    
    for select_ in tree_paths:

        tree_list = torch.load(tree_paths[select_])
        if 'scc' in select_:
            trees_filtered = [x for x in tree_list if len(str.split(x[1])) < 20]
        elif 'libri' in select_:
            trees_filtered = [x for x in tree_list if len(str.split(x[1])) < 52]
        st = time.time()
        # df = calculate_kernel(tree_list)
        # df.to_csv(select_[:-4]+'kernel.csv')
        tree_kernel = calculate_kernel(trees_filtered)
        torch.save(tree_kernel, select_[:-4]+'kernel.pt')
        et = time.time()
        print('script ran for', et-st, 'seconds for', select_[:-4])