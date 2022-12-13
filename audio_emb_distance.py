import os

import numpy as np
import torch
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances
from torchmetrics.functional import pairwise_cosine_similarity
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
import ursa.util as U


def audio_pairwise_distance_calc(dataset):
    pairwise_distance_container = {}
    for layer in range(len(dataset[0][0])):
        print('pulling audio embedding from layer', layer)
        audio_emb = [x[0][layer][:-2] for x in dataset]
        # layer_distance = pairwise_distances(audio_emb, metric='cosine', n_jobs=-1)
        # layer_distance = U.pairwise(cosine,audio_emb, parallel=True)
        tensor = torch.from_numpy(np.array(audio_emb).astype('float16'))
        layer_similarity = pairwise_cosine_similarity(tensor.to(device)).cpu()
        layer_similarity[layer_similarity == 0] = 1
        # pairwise_distance_container[layer] = layer_distance
        pairwise_distance_container[layer] = U.triu(layer_similarity)
    return pairwise_distance_container

if __name__ == "__main__":
    datasets_path = '/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings'
    dataset_files = [os.path.join(datasets_path,x) for x in os.listdir(datasets_path)]
    dataset_names = [os.path.basename(x[:-13]) for x in dataset_files]
    dataset_lookup = dict(zip(dataset_names, dataset_files))

    # distance_lookup = {}

    for i in dataset_lookup:
        save_file = i+'_pairwise_distance.pt'
        if os.path.isfile(save_file):
            print(f"{save_file} already exists! skipping for now")

        else:
            print('calculating pairwise distance of audio emb from', i)
            dataset = torch.load(dataset_lookup[i])
            distance_calculated = audio_pairwise_distance_calc(dataset)

            distance_lookup = distance_calculated

        
            torch.save(distance_lookup, save_file)

            # break



