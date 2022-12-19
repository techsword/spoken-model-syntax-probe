import random
import os

import torch


import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import time

from ursa.kernel import Kernel

tree_paths = {  "scc_trees_path": '/home/gshen/work_dir/spoken-model-syntax-probe/scc_generated_trees.pt', 
                "libri_trees_path": '/home/gshen/work_dir/spoken-model-syntax-probe/libri_generated_trees.pt'}




def calculate_kernel(tree_list, normalization = True, alpha = 0.5):
    K = Kernel(alpha=alpha)
    tree_kernel = []
    for i in range(0, len(tree_list), 2):
        tree_1, tree_2 = tree_list[i][0][0], tree_list[i+1][0][0]
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


def main(seed = 42, alpha = 0.5):
    random.seed(seed)
    for select_ in tree_paths:
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
            # df = calculate_kernel(tree_list)
            # df.to_csv(select_[:-4]+'kernel.csv')
            random.shuffle(trees_filtered)
            tree_kernel = calculate_kernel(trees_filtered, alpha = alpha)
            torch.save(tree_kernel, save_file)
            et = time.time()
            print('script ran for', et-st, 'seconds for', select_[:-5])

if __name__ == "__main__":
    main(seed = 666)    


