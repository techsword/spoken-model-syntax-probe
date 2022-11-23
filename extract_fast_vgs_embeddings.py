import sys
import os
import pickle
model_dir = "/home/gshen/work_dir/spoken-model-syntax-probe/fast_vgs_family/model_path"
split = ['fast_vgs_family/model_path/fast-vgs-coco', 'fast_vgs_family/model_path/fast-vgs-plus-coco']
import torch
import numpy as np
from fast_vgs_family.models import fast_vgs, w2v2_model


from custom_classes import Corpus

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from itertools import islice

sr = 16000

def load_fast_vgs_model(model_path):
    # load args
    with open(f"{model_path}/args.pkl", "rb") as f:
        args = pickle.load(f)
    # load weights
    weights = torch.load(os.path.join(model_path, "best_bundle.pth"))

    # # if want to use the entire model for e.g. speech-image retrieval (need to first follow section 3 below)
    # dual_encoder = fast_vgs.DualEncoder(args)
    # cross_encoder = fast_vgs.CrossEncoder(args)
    # dual_encoder.load_state_dict(weights['dual_encoder'])
    # cross_encoder.load_state_dict(weights['cross_encoder'])

    if 'plus' in model_path:
        args_dict = vars(args)
        args_dict['trim_mask'] = False


    # if only want to use the audio branch for e.g. feature extraction for speech downstream tasks
    # if you are loading fast-vgs features, it will say that weights of layer 8-11 (0-based) are not seed_dir, that's fine, because fast-vgs only has first 8 layers (i.e. layer 0-7) of w2v2 model, last four layers will be randomly initialized layers
    model = w2v2_model.Wav2Vec2Model_cls(args)
    model.carefully_load_state_dict(weights['dual_encoder']) # will filter out weights that don't belong to w2v2

    return model


def generating_features(dataset, model, limit = None):
    feat_list = []
    lab_list = []
    annot_list = []
    wav_path_list = []
    iter = 0

    for waveform, annot, depth, path in islice(dataset,limit):
        wordcount = len(str.split(annot))
        audio_len = len(waveform[-1])/sr
        # if depth < 13 and depth > 5:
        lab_list.append(depth)
        annot_list.append(annot)
        wav_path_list.append(path)
        model = model.to(device)
        with torch.inference_mode():
            features_dict = model(source=waveform.to(device), padding_mask=None, mask=False, superb=True, tgt_layer=7)
            features = features_dict['hidden_states']
            features = [torch.mean(x.cpu(),dim=1).squeeze().numpy() for x in features]
            features = [np.append(layer,[audio_len]) for layer in features]
            features = [np.append(layer,[wordcount]) for layer in features]
            # feat_list.append(torch.mean(features,dim=1).squeeze().numpy())
            feat_list.append(features)


    
    print(f"there are {len(lab_list)} in the extracted dataset, each tensor is {features[0].shape}, the max tree depth is {max(lab_list)} and the min is {min(lab_list)}")
    return list(zip(feat_list, lab_list,annot_list,wav_path_list))

if __name__ == '__main__':
    dataset_select = 'librispeech'
    fast_vgs = load_fast_vgs_model(split[0])


    if dataset_select == 'spokencoco':

        root_dir='/home/gshen/SpokenCOCO/'
        csv_file = 'spokencoco_val.csv'
        dataset = Corpus(csv_file, root_dir = root_dir)
        
        scc_extracted = generating_features(dataset, fast_vgs)
        torch.save(scc_extracted, 'fast_vgs_spokencoco_val_extracted.pt')

    elif dataset_select == 'librispeech':
        libri_split = 'train-clean-100'
        librispeech_root = '/home/gshen/work_dir/librispeech-train/'
        dataset = Corpus('librispeech_'+libri_split+'.csv', os.path.join(librispeech_root, libri_split))
        libri_extracted = generating_features(dataset, fast_vgs)
        torch.save(libri_extracted, 'fast_vgs_librispeech_train_extracted.pt')

