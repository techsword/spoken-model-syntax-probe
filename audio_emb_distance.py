import os

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
import ursa.util as U
from scipy.spatial.distance import cosine


def audio_pairwise_distance_calc(dataset):
    pairwise_distance_container = {}
    for layer in range(len(dataset[0][0])):
        print('pulling audio embedding from layer', layer)
        audio_emb = [x[0][layer][:-2] for x in dataset]
        # layer_distance = pairwise_distances(audio_emb, metric='cosine', n_jobs=-1)
        layer_distance = U.pairwise(cosine,audio_emb, parallel=True)
        pairwise_distance_container[layer] = layer_distance
    return pairwise_distance_container

if __name__ == "__main__":
    datasets_path = '/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings'
    dataset_files = [os.path.join(datasets_path,x) for x in os.listdir(datasets_path)]
    dataset_names = [os.path.basename(x[:-13]) for x in dataset_files]
    dataset_lookup = dict(zip(dataset_names, dataset_files))

    # distance_lookup = {}

    for i in dataset_lookup:
        print('calculating pairwise distance of audio emb from', i)
        dataset = torch.load(dataset_lookup[i])
        distance_calculated = audio_pairwise_distance_calc(dataset)

        distance_lookup = distance_calculated

    
        torch.save(distance_lookup, i+'_pairwise_distance.pt')


