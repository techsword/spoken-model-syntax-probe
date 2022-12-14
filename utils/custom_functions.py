import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


sr = 16000


def loading_fairseq_model(model_file):
    import fairseq

    from torchaudio.models.wav2vec2.utils import import_fairseq_model
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
    original = model[0]
    imported = import_fairseq_model(original)
    return imported

def loading_huggingface_model(hf_model):
    from torchaudio.models.wav2vec2.utils import import_huggingface_model
    model = import_huggingface_model(hf_model)
    return model

def walk_librispeech_dirs(librispeech_root, libri_split):
    libri_split_path = os.path.join(librispeech_root,libri_split)
    filelist = []
    for root, dir, file in os.walk(libri_split_path):
        filelist.append(file)
    import itertools
    flatlist = list(itertools.chain(*filelist))
    txtlist = [os.path.join(*[re.split(r'\.|-', x)[0],re.split(r'\.|-', x)[1],x]) for x in flatlist if 'txt' in x]
    flaclist = [os.path.join(*[re.split(r'\.|-', x)[0],re.split(r'\.|-', x)[1],x]) for x in flatlist if 'flac' in x]
    df = pd.DataFrame(flaclist)
    # df['dirname'] = df[0].apply(lambda x : os.path.split(x)[0])
    df['fileid'] = df[0].apply(lambda x : os.path.split(x)[1])
    df['fileid'] = df['fileid'].str.replace('.flac', '')
    df = df.rename({0:'fullpath'},axis=1)
    df_txt = pd.concat([pd.read_csv(os.path.join(libri_split_path,item), header=None) for item in txtlist],ignore_index=True)
    df_txt['fileid'] = df_txt[0].str.extract('(\d+-\d+-\d+)')
    df_txt['sent'] = df_txt[0].str.replace(r'(\d+-\d+-\d+\s)', '')
    df_txt = df_txt.drop([0], axis=1)
    merge_df = pd.merge(df,df_txt, on = 'fileid')
    return merge_df
    
def make_bow(doc):
    
    doc = list(map(str.lower, doc))
    unique_words = set(' '.join(doc).split())
    print(f'there are {len(unique_words)} unique words')
    index_dict = {}
    for ind, i in enumerate(sorted(unique_words)):
        index_dict[i] = ind
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    count_occurrences = cv.fit_transform(doc)
    return count_occurrences.toarray()

def generating_features(dataset, model):
    feat_list = []
    lab_list = []
    annot_list = []
    wav_path_list = []
    for waveform, annot, depth, path in dataset:
        wordcount = len(str.split(annot))
        # if depth < 13 and depth > 5:
        lab_list.append(depth)
        annot_list.append(annot)
        wav_path_list.append(path)
        with torch.inference_mode():
            features, _ = model.to(device).extract_features(waveform.to(device))
            # print(features)
            audio_len = len(waveform[-1])/sr
            features = [torch.mean(x.cpu(),dim=1).squeeze().numpy() for x in features]
            features = [np.append(layer,[audio_len]) for layer in features]
            features = [np.append(layer,[wordcount]) for layer in features]
            # feat_list.append(torch.mean(features,dim=1).squeeze().numpy())
            feat_list.append(features)

    
    print(f"there are {len(lab_list)} in the extracted dataset, each tensor is {features[0].shape}, the max tree depth is {max(lab_list)} and the min is {min(lab_list)}")
    return list(zip(feat_list, lab_list,annot_list,wav_path_list))

def read_json_save_csv(json_path):
    '''Reading the SpokenCOCO json file and extracting the text and wav file pairs'''
    with open(json_path) as json_file:
        data = json.load(json_file)
    text = []
    wav = []
    for image in data['data']:
        for caption in image['captions']:
            text.append(caption['text'])
            wav.append(caption['wav'])


    '''
    Using pandas to combine the two lists extracted in the cells above and 
    save the resulting dataframe into a csv file
    '''
    dict_spokencoco = dict(zip(text,wav))
    spokencoco_val = pd.Series(dict_spokencoco, name = 'wav')
    spokencoco_val.index.name = 'text'
    spokencoco_val = spokencoco_val.reset_index()
    column_titles = ['wav', 'text']
    spokencoco_val = spokencoco_val.reindex(columns = column_titles)
    return spokencoco_val




# def generating_features(dataset, model_file):
    # feat_list = []
    # lab_list = []
    # annot_list = []
    # wav_path_list = []
    # model = loading_fairseq_model(model_file=model_file)
    # import stanza
    # nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    # for waveform, annot, path in dataset:
    #     doc = nlp(annot)
    #     for sent in doc.sentences:
    #         depth = sent.constituency.depth()
    #         wordcount = len(sent.words)
    #         if depth < 13 and depth > 5:
    #             lab_list.append(depth)
    #             annot_list.append(annot)
    #             wav_path_list.append(path)
    #             with torch.inference_mode():
    #                 features, _ = model.to(device).extract_features(waveform.to(device))
    #                 # print(features)
    #                 audio_len = len(waveform[-1])/sr
    #                 features = [torch.mean(x.cpu(),dim=1).squeeze().numpy() for x in features]
    #                 features = [np.append(layer,[audio_len]) for layer in features]
    #                 features = [np.append(layer,[wordcount]) for layer in features]
    #                 # feat_list.append(torch.mean(features,dim=1).squeeze().numpy())
    #                 feat_list.append(features)

    
    # print(f"there are {len(lab_list)} in the extracted dataset, each tensor is {features[0].shape}, the max tree depth is {max(lab_list)} and the min is {min(lab_list)}")
    # return list(zip(feat_list, lab_list,annot_list,wav_path_list))



def get_weird_sents(corpus_csv, root_dir):
    from custom_classes import Corpus
    corpus_ = Corpus(corpus_csv, root_dir)
    weird_sents = [x for x in [corpus_.get_depth(i) for i in range(len(corpus_))] if x[1] > len(str.split(x[0]))+2]
    print(f'there are total {len(weird_sents)} sentences')
    print(weird_sents)

def generate_tree(dataset, num_entries = None):
    import stanza
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    from nltk.tree import Tree

    text = [x[-2] for x in dataset[:num_entries]]

    tree_list = [str(nlp(sent).sentences[0].constituency) for sent in text]
    nltk_tree_list = [Tree.fromstring(x) for x in tree_list]

    return list(zip((nltk_tree_list,text)))

def recover_from_triu(matrices):

    size_X = matrices[-1]
    X = np.zeros((size_X,size_X))
    full_matrices = []
    for v in matrices[:-1]:
        X[np.triu_indices(X.shape[0], k = 1)] = v
        X = X + X.T
        full_matrices.append(X)
    return full_matrices