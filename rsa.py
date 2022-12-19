import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import os

from scipy.stats import pearsonr

tree_kernel_path = '/home/gshen/work_dir/spoken-model-syntax-probe/tree_kernel'
kernel_files = [os.path.join(tree_kernel_path,x) for x in os.listdir(tree_kernel_path) if '_kernel' in x]
kernel_lookup = dict(zip([os.path.basename(x)[:-10] for x in kernel_files], kernel_files))

pairwise_distance_path = '/home/gshen/work_dir/spoken-model-syntax-probe/pairwise_distances'

pairwise_distance_files = [os.path.join(pairwise_distance_path,x) for x in os.listdir(pairwise_distance_path) if 'full' in x]

def main(alpha = 0.5, seed = 42):
    alpha = str(alpha)
    seed = str(seed)
    for x in pairwise_distance_files:

        # print(x)
        
        if 'librispeech' in x:
            kernel_file = kernel_lookup['libri_trees_'+alpha+'_'+seed]

        elif 'spokencoco' in x:
            kernel_file = kernel_lookup['scc_trees_'+alpha+'_'+seed]
        
        kernel_set = np.array(torch.load(kernel_file), dtype=object)
        kernel = np.array(kernel_set[:,0])
        kernel_pairs = np.array(kernel_set[:,1:])

        calculated_distances = torch.load(x)
            
        for layer in range(len(calculated_distances[:-1])):
            # audio_distances = [calculated_distances[layer][x[0],x[1]] for x in kernel_pairs]
            audio_distances = np.array([calculated_distances[layer][x[0],x[1]] for x in kernel_pairs])
            
            r_score = pearsonr(kernel, audio_distances)
            name_list = str.split(os.path.basename(x),sep='_')[:-3]
            modelname = '_'.join(name_list[:-2])
            datasetname = '_'.join(name_list[-2:])
            print((modelname, datasetname, layer, alpha) + r_score)

if __name__ == "__main__":
    main(seed=666)