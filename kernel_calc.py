import os
import random

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import time

from nltk.tree import Tree
from ursa.kernel import Kernel

tree_paths = {  "scc_trees_path": '/home/gshen/work_dir/spoken-model-syntax-probe/scc_generated_trees.pt', 
                "libri_trees_path": '/home/gshen/work_dir/spoken-model-syntax-probe/libri_generated_trees.pt'}


def delex(n, leaf="X"):
    if isinstance(n, str): 
        return leaf 
    else: 
        return Tree(n.label(), [ delex(c) for c in n[:] ])


def calculate_kernel_regular_pair(tree_list, normalization = True, alpha = 0.5, delex_option = False):
    K = Kernel(alpha=alpha)
    tree_kernel = []
    for i in range(0, len(tree_list), 2):
        if delex_option:
            tree_1, tree_2 = tree_list[i][0][0], tree_list[i+1][0][0]
        else:
            tree_1, tree_2 = delex(tree_list[i][0][0]), delex(tree_list[i+1][0][0])
        kern = K(tree_1, tree_2)
        if normalization == True:
            # Normalize the kernels calculated 
            denom = (K(tree_1, tree_1)*K(tree_2,tree_2))**0.5
        else:
            denom = 1

        normed_kern = kern/denom

        tree_kernel.append((normed_kern,tree_list[i][-1],tree_list[i+1][-1]))
        # print(i)
    return tree_kernel


def generate_tree_kernel_for_random_pairs(seed = 42, alpha = 0.5, delex_option = False):
    random.seed(seed)
    for select_ in tree_paths:
        if delex_option:
            save_file = os.path.join('tree_kernel', select_[:-4]+str(alpha)+'_'+str(seed)+'_'+'delexed'+'_'+'kernel.pt')
        else:
            save_file = os.path.join('tree_kernel', select_[:-4]+str(alpha)+'_'+str(seed)+'_'+'kernel.pt')

        if os.path.isfile(save_file):
            print(save_file + ' already exists! skipping')
        else:
            print(f"generating {os.path.basename(save_file)}")
            tree_list = torch.load(tree_paths[select_])
            if 'scc' in select_:
                trees_filtered = [x for x in tree_list if len(str.split(x[1])) < 20]
            elif 'libri' in select_:
                trees_filtered = [x for x in tree_list if len(str.split(x[1])) < 52]
            trees_filtered = [[item, i] for i, item in enumerate(trees_filtered)]
            st = time.time()
            random.shuffle(trees_filtered)
            tree_kernel = calculate_kernel_regular_pair(trees_filtered, alpha = alpha)
            torch.save(tree_kernel, save_file)
            et = time.time()
            print('script ran for', et-st, 'seconds for', select_[:-5])

def generate_tree_kernel_for_regress(seed =42, alpha = 0.5, delex_option = True, num_anchors = 5):
    random.seed(seed)
    for select_ in tree_paths:
        if delex_option:
            save_file = os.path.join('tree_kernel_regress', select_[:-4]+str(alpha)+'_'+str(seed)+'_'+'delexed'+'_'+'kernel.pt')
        else:
            save_file = os.path.join('tree_kernel_regress', select_[:-4]+str(alpha)+'_'+str(seed)+'_'+'kernel.pt')
        if os.path.isfile(save_file):
            print(save_file + ' already exists! skipping')
        else:
            print(f"generating {os.path.basename(save_file)}")
            tree_list = torch.load(tree_paths[select_])
            if 'scc' in select_:
                trees_filtered = [x for x in tree_list if len(str.split(x[1])) < 20]
            elif 'libri' in select_:
                trees_filtered = [x for x in tree_list if len(str.split(x[1])) < 52]
            trees_filtered = [[item, i] for i, item in enumerate(trees_filtered)]
            st = time.time()
            random.shuffle(trees_filtered)
            # tree_kernel = calculate_kernel_regular_pair(trees_filtered, alpha = alpha)
            ref_pts = random.choices(trees_filtered, k = num_anchors)
            for ref_pt in ref_pts:
                ref_pt_kern = K(ref_pt, ref_pt)
                for tree_2 in trees_filtered:
                    kern = K(ref_pt, tree_2)
                    # Normalize the kernels calculated 
                    denom = (ref_pt_kern*K(tree_2,tree_2))**0.5

                    normed_kern = kern/denom
            
            # torch.save(tree_kernel, save_file)

            et = time.time()
            print('script ran for', et-st, 'seconds for', select_[:-5])


if __name__ == "__main__":
    # generate_tree_kernel_for_random_pairs(seed = 42)
    # generate_tree_kernel_for_random_pairs(seed = 666)    
    # generate_tree_kernel_for_random_pairs(seed = 19)
    # generate_tree_kernel_for_random_pairs(seed = 2022)
    generate_tree_kernel_for_random_pairs(seed = 42, delex_option=True)
    generate_tree_kernel_for_random_pairs(seed = 666, delex_option=True)   
    generate_tree_kernel_for_random_pairs(seed = 19, delex_option=True)
    generate_tree_kernel_for_random_pairs(seed = 2022, delex_option=True)


