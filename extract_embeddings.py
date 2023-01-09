import argparse
import os
import pickle
from itertools import islice

import numpy as np
import torch
from torchsummary import summary

from fast_vgs_family.models import fast_vgs, w2v2_model
from utils.custom_classes import Corpus
from utils.custom_functions import (generating_features, read_json_save_csv,
                                    walk_librispeech_dirs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sr = 16000

# librispeech settings
librispeech_root = '/home/gshen/work_dir/librispeech-train/'# 'LibriSpeech/
libri_split = 'train-clean-100' # 'test-clean'

# spokencoco settings

json_path = '/home/gshen/SpokenCOCO/SpokenCOCO_val.json'
root_dir='/home/gshen/SpokenCOCO/'
csv_file = 'spokencoco_val.csv'


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


def fast_vgs_feat_gen(dataset, model, limit = None):
    feat_list = []
    lab_list = []
    annot_list = []
    wav_path_list = []
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

    print(f"there are {len(lab_list)} in the extracted dataset, the total number of layers is {len(feat_list)}, each tensor is {features[0].shape}, the max tree depth is {max(lab_list)} and the min is {min(lab_list)}")
    return list(zip(feat_list, lab_list,annot_list,wav_path_list))

if __name__ == '__main__':    
    
    # setting the model to extract embeddings and the output filename
    model_dict = {'hubert': '/home/gshen/work_dir/spoken-model-syntax-probe/hubert_base_ls960.pt', 'wav2vec':'/home/gshen/work_dir/wav2vec_small.pt'}
    save_path = 'extracted_embeddings'

    parser = argparse.ArgumentParser(description='control what model and which dataset you are using for embedding extraction')

    parser.add_argument('--model',
                    type=str, default = 'hubert', metavar='model',
                    help="choose the model used to extract embeddings, default is hubert. options: hubert, wav2vec, random, fast-vgs, fast-vgs-plus, asr"
    )
    parser.add_argument('--corpus',
                    type=str, default = 'spokencoco', metavar='corpus',
                    help="choose the corpus to extract embeddings from, default is spokencoco. options: spokencoco, librispeech"
    )
    args = parser.parse_args()

    if 'fast-vgs' not in args.model:
        if args.model == 'asr':
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model
            MODEL_ID = "jonatasgrosman/wav2vec2-large-english"
            # processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
            hf_model = Wav2Vec2Model.from_pretrained(MODEL_ID)
            saved_file = 'wav2vec_large_en_' + args.corpus + '_extracted.pt'
            from utils.custom_functions import loading_huggingface_model
            model = loading_huggingface_model(hf_model)

        elif args.model == 'random':
            from transformers import Wav2Vec2Config, Wav2Vec2Model

            # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
            configuration = Wav2Vec2Config()
            # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
            hf_model = Wav2Vec2Model(configuration)
            # Accessing the model configuration
            configuration = hf_model.config
            saved_file = 'wav2vec_random_' + args.corpus + '_extracted.pt'
            from utils.custom_functions import loading_huggingface_model
            model = loading_huggingface_model(hf_model)

        else: #args.model != 'random' or 'asr':
            model_file = model_dict[args.model]
            saved_file = os.path.basename(model_file[:-3]) + '_' + args.corpus + '_extracted' + '.pt'
            from utils.custom_functions import loading_fairseq_model
            model = loading_fairseq_model(model_file)

        if args.corpus == 'spokencoco':
            # extract embeddings with the model defined above from spokencoco corpus
            if os.path.isfile(csv_file) == False:
                # making sure the csv file is there, otherwise create the csv file
                print(f"{csv_file} not found, creating from {json_path}")
                spokencoco_df = read_json_save_csv(json_path)
                spokencoco_df.to_csv(csv_file, header=None, index = None)
            print(f"generating features")
            spokencoco = Corpus(csv_file, root_dir = root_dir)
            spokencoco_extracted = generating_features(spokencoco, model)
            print(f"saving the extracted embeddings to {saved_file}")
            torch.save(spokencoco_extracted, os.path.join(save_path, saved_file))

        elif args.corpus == "librispeech":
            # extract embeddings with the model defined above from librispeech corpus
            if os.path.isfile('librispeech_'+libri_split+'.csv') == False:
                # making sure the csv file is there, otherwise create the csv file
                print(f"{'librispeech_'+libri_split+'.csv'} not found, creating from {os.path.join(librispeech_root, libri_split)}")
                librispeech_dataset_df = walk_librispeech_dirs(librispeech_root=librispeech_root, libri_split=libri_split)
                librispeech_dataset_df.to_csv('librispeech_'+libri_split+'.csv', index=False)
            print(f"generating features")
            libri_ds = Corpus('librispeech_'+libri_split+'.csv', os.path.join(librispeech_root, libri_split))
            libri_extracted = generating_features(libri_ds, model)
            print(f"saving the extracted embeddings to {saved_file}")
            torch.save(libri_extracted, os.path.join(save_path,saved_file))

    elif 'fast-vgs' in args.model:
        

        model_dir = "/home/gshen/work_dir/spoken-model-syntax-probe/fast_vgs_family/model_path"
        split = {'fast-vgs':'fast_vgs_family/model_path/fast-vgs-coco', 'fast-vgs-plus':'fast_vgs_family/model_path/fast-vgs-plus-coco'}

        fast_vgs = load_fast_vgs_model(split[args.model])
        print(f'finished loading model  {args.model}')
        if args.corpus == 'spokencoco':

            root_dir='/home/gshen/SpokenCOCO/'
            csv_file = 'spokencoco_val.csv'
            dataset = Corpus(csv_file, root_dir = root_dir)
            scc_extracted = fast_vgs_feat_gen(dataset, fast_vgs)
            torch.save(scc_extracted, os.path.join(save_path, args.model+'_spokencoco_val_extracted.pt'))

        elif args.corpus == 'librispeech':
            libri_split = 'train-clean-100'
            librispeech_root = '/home/gshen/work_dir/librispeech-train/'
            dataset = Corpus('librispeech_'+libri_split+'.csv', os.path.join(librispeech_root, libri_split))
            libri_extracted = fast_vgs_feat_gen(dataset, fast_vgs)
            torch.save(libri_extracted, os.path.join(save_path, args.model+'_librispeech_train_extracted.pt'))
        print(f'finished extracting embeddings from {args.corpus} with {args.model}')
    
