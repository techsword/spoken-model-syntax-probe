import os

import numpy as np
import torch
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances
from torchmetrics.functional import pairwise_cosine_similarity
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
import ursa.util as U


def pairwise_distance_calc(dataset):
    pairwise_distance_container = []
    for layer in range(len(dataset[0][0])):
        print('pulling embedding from layer', layer)
        embeddings = [x[0][layer][:-2] for x in dataset]
        # layer_distance = pairwise_distances(embeddings, metric='cosine', n_jobs=-1)
        # layer_distance = U.pairwise(cosine,embeddings, parallel=True)
        tensor = torch.from_numpy(np.array(embeddings).astype('float'))
        layer_similarity_ = pairwise_cosine_similarity(tensor.to(device)).cpu()
        # layer_similarity[layer_similarity == 0] = 1
        # layer_similarity = U.triu(layer_similarity_).clone()
        # layer_similarity = torch.diagonal(layer_similarity_,1)[::2].clone()
        pairwise_distance_container.append(layer_similarity_.clone())
    pairwise_distance_container.append(len(dataset))

    return pairwise_distance_container

if __name__ == "__main__":
    datasets_path = '/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings'
    dataset_files = [os.path.join(datasets_path,x) for x in os.listdir(datasets_path)]
    dataset_names = [os.path.basename(x[:-13]) for x in dataset_files]
    dataset_lookup = dict(zip(dataset_names, dataset_files))


    for i in dataset_lookup:
        save_file = 'pairwise_distances/'+i+'_pairwise_distance_full.pt'
        if os.path.isfile(save_file):
            print(f"{save_file} already exists! skipping for now")

        else:
            print('calculating pairwise distance of embeddings from', i)
            dataset = torch.load(dataset_lookup[i])
            if 'libri' in i:
                dataset = [x for x in dataset if len(str.split(x[2])) < 52]

            elif 'spokencoco' in i:
                dataset = [x for x in dataset if len(str.split(x[2])) < 20]
            distance_calculated = pairwise_distance_calc(dataset)

            torch.save(distance_calculated, save_file)

            # break



