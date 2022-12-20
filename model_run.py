import argparse
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import stanza
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils.custom_functions import make_bow

# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

def load_model(model_name):
    if model_name == 'logreg':
        model = LogisticRegression(solver = 'saga',multi_class='auto', max_iter=10000)
    elif model_name == 'ridge':
        model = RidgeCV()
    elif model_name == "svm":
        model = SVC(kernel = 'linear', C = 1,max_iter=10000)
    return model


def model_training(X,y, model):
        if X[0].size == 1:
            X_train, X_test, y_train, y_test = train_test_split(np.array(X).reshape(-1,1), y, random_state = 42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

        model.fit(X_train, y_train)
        if not str(type(model)) == "<class 'sklearn.linear_model._ridge.RidgeCV'>":
            y_pred = model.predict(X_test)
            r2score = r2_score(y_test,y_pred)
            mse = mean_squared_error(y_test,y_pred)
            acc = accuracy_score(y_test,y_pred)
            return r2score, mse, acc
        elif str(type(model)) == "<class 'sklearn.linear_model._ridge.RidgeCV'>":
            model_alpha = model.alpha_
            r2score = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test,y_pred)
            return r2score, mse, model_alpha

def baseline(embeddings, y, model):
    wordcount_ = [x[-1] for x in embeddings]
    audio_len_ = [x[-2] for x in embeddings]
    # y = np.array(labels)
    scoring = []
    for i, X in enumerate([wordcount_, audio_len_]):
        lookup = {0: "WC-base", 1: "AL-base"}
        result = list(model_training(X,y, model)) + [lookup[i]]
        scoring.append(result)
    return scoring


def iter_layers(embeddings, y, lay, model, combination = False):

    # y = np.array(labels)
    scoring = []
    print(f"running model on layer {lay}")
    # just audio
    if len(embeddings[0]) == 770:
        embeddings_ = [np.delete(x, [-2,-1]) for x in embeddings]
        scoring += [[lay] + list(model_training(embeddings_, y, model)) + ['EMB']]
        if combination:
            # audio + audiolen
            X = [np.delete(x, -1) for x in embeddings]
            scoring += [[lay] + list(model_training(X, y, model)) + ['AL']]
            # audio+wordcount
            X = [np.delete(x, -2) for x in embeddings]
            scoring += [[lay] + list(model_training(X, y, model)) + ['WC']]
            # everything
            X = [x for x in embeddings]
            scoring += [[lay] + list(model_training(X, y, model)) + ['EMB+AL+WC']]
    
    elif len(embeddings[0]) == 768:

        scoring += [[lay] + list(model_training(embeddings, y, model)) + ['EMB']]

    return scoring




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='controlling number of data and layer used in model training')
    # parser.add_argument('--layer',
    #                 type = int, default = 0, metavar='layer',
    #                 help = 'add the layer number to train the regression model on'
    # )
    parser.add_argument('--num_data',
                    type=int, default = None, metavar='num-data',
                    help='add the number of elements to include from the dataset')
    parser.add_argument('--baseline',
                    type=bool, default = False, metavar='baseline',
                    help='decide on if you want to measure the baseline ')
    parser.add_argument('--model',
                    type=bool, default = False, metavar='model',
                    help='actually training the model, default = False')
    parser.add_argument('--modelname',
                    type=str, default = 'ridge', metavar='modelname',
                    help="choose the sklearn models to run here, default is ridge with cross validation. options: ridge, svm, logreg"
    )
    parser.add_argument('--dataset',
                    type=str, default = 'scc-hubert', metavar='dataset',
                    help="choose embeddings extracted from which dataset to run here, default is hubert. options: wav2vec, hubert"
    )



    args = parser.parse_args()
    print(f"start loading embedding for model")
    path_to_extracted_embeddings = '/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings'
    audio_model_names = {'hubert'   :'hubert_base_ls960',
                        'fvgs'      : 'fast_vgs',
                        'fvgs+'     :'fast_vgs_plus',
                        'wav2vec'   : 'wav2vec_small',
                        'wav2vec-random': 'wav2vec_random',
                        'deberta': 'deberta'}

    corpora_names = {'libri': 'librispeech_train',
                    'scc'   : 'spokencoco_val'}

    dataset_dict = {"scc-hubert":   os.path.join(path_to_extracted_embeddings, audio_model_names['hubert']+'_'+corpora_names['scc'] +"_extracted.pt" ),
                    'scc-wav2vec' :os.path.join(path_to_extracted_embeddings, audio_model_names['wav2vec']+'_'+corpora_names['scc'] +"_extracted.pt" ),
                    'libri-wav2vec':os.path.join(path_to_extracted_embeddings, audio_model_names['wav2vec']+'_'+corpora_names['libri'] +"_extracted.pt" ),
                    'scc-wav2vec-random' :os.path.join(path_to_extracted_embeddings, audio_model_names['wav2vec-random']+'_'+corpora_names['scc'] +"_extracted.pt" ),
                    'libri-wav2vec-random':os.path.join(path_to_extracted_embeddings, audio_model_names['wav2vec-random']+'_'+corpora_names['libri'] +"_extracted.pt" ),
                    'libri-hubert': os.path.join(path_to_extracted_embeddings, audio_model_names['hubert']+'_'+corpora_names['libri'] +"_extracted.pt" ),
                    'scc-fast-vgs': os.path.join(path_to_extracted_embeddings, audio_model_names['fvgs']+'_'+corpora_names['scc'] +"_extracted.pt" ),
                    'libri-fast-vgs': os.path.join(path_to_extracted_embeddings, audio_model_names['fvgs']+'_'+corpora_names['libri'] +"_extracted.pt" ),
                    'scc-fast-vgs-plus': os.path.join(path_to_extracted_embeddings, audio_model_names['fvgs+']+'_'+corpora_names['scc'] +"_extracted.pt" ),
                    'libri-fast-vgs-plus': os.path.join(path_to_extracted_embeddings, audio_model_names['fvgs+']+'_'+corpora_names['libri'] +"_extracted.pt" ),
                    'libri-deberta': os.path.join(path_to_extracted_embeddings, audio_model_names['deberta']+'_'+corpora_names['libri'] +"_sentemb.pt"),
                    'scc-deberta': os.path.join(path_to_extracted_embeddings, audio_model_names['deberta']+'_'+corpora_names['scc'] +"_sentemb.pt")
                    }
    dataset = dataset_dict[args.dataset]
    loaded_embeddings, labels, annot, wav = zip(*torch.load(dataset))
    BOW_array = make_bow(annot)
    num_layers = len(loaded_embeddings[0])
    labels = np.array(labels[:args.num_data])


    if args.baseline == True:
        print(f'measuring baseline')
        embeddings = [x[0] for x in loaded_embeddings][:args.num_data]
        wordcount_ = [x[-1] for x in embeddings]
        audio_len_ = [x[-2] for x in embeddings]
        # baseline_score = baseline(embeddings,labels, load_model(args.modelname))
        print(f"measuring BOW baseline")
        WC_baseline = model_training(wordcount_,labels,load_model(args.modelname))
        BOW_baseline = model_training(BOW_array,labels,load_model(args.modelname))
        AL_baseline = model_training(audio_len_,labels,load_model(args.modelname))
        scoring = []
        for layer in range(num_layers):
            scoring += [[layer] + list(BOW_baseline)+['BOW-baseline']]
            scoring += [[layer] + list(WC_baseline)+['WC-baseline']]
            scoring += [[layer] + list(AL_baseline)+['AL-baseline']]
            # scoring += [[layer] + x for x in list(baseline_score)]
        print(f'the baseline score is \n {scoring}')
    
    elif args.model == True:
        for i in range(num_layers):
            embeddings = [x[i] for x in loaded_embeddings][:args.num_data]
            scoring = []
            if len(embeddings[0]) == 770:
                results = iter_layers(embeddings, labels, i, load_model(args.modelname), True)
            elif len(embeddings[0]) == 768:
                results = iter_layers(embeddings, labels, i, load_model(args.modelname), False)
            scoring += results
            bigarray = np.concatenate((BOW_array, embeddings), axis=1)
            bow_results = iter_layers(bigarray, labels, i, load_model(args.modelname))
            scoring += [x + ['BOW'] for x in bow_results]
            print(scoring)



'''
running model on layer 0
running model on layer 0
[[[0, 0.10994011180282237, 1.937616249729601, 1.0, 'EMB']], [[0, 0.11200118659093505, 1.9331293920986565, 1.0, 'AL']], [[0, 0.22254394759880702, 1.6924832818097515, 10.0, 'WC']], [[0, 0.2246420942851246, 1.6879157204943442, 10.0, 'EMB+AL+WC']], [[0, 0.5347980550919789, 1.0127215705511161, 10.0, 'EMB'], 'BOW'], [[0, 0.5348159873750946, 1.0126825328598092, 10.0, 'AL'], 'BOW'], [[0, 0.5436455813190888, 0.9934609445922488, 10.0, 'WC'], 'BOW'], [[0, 0.5439752314488793, 0.9927433126028988, 10.0, 'EMB+AL+WC'], 'BOW']]

real    15m8.383s
user    170m24.209s
sys     73m24.798s

running model on layer 1
running model on layer 1
[[1, 0.12646668855790233, 1.90163871147888, 1.0, 'EMB'], [1, 0.1287335271289527, 1.8967039162937183, 1.0, 'AL'], [1, 0.23106945912946097, 1.6739236658804102, 10.0, 'WC'], [1, 0.2330524898024361, 1.669606706418894, 10.0, 'EMB+AL+WC'], [1, 0.5346199199601538, 1.0131093619016183, 10.0, 'EMB', 'BOW'], [1, 0.5344856897453609, 1.013401574424434, 10.0, 'AL', 'BOW'], [1, 0.542267787036907, 0.9964603344370685, 10.0, 'WC', 'BOW'], [1, 0.5425732615527529, 0.9957953315607112, 10.0, 'EMB+AL+WC', 'BOW']]

real    13m51.015s
user    185m23.259s
sys     74m58.015s


'''