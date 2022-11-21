import torch
import numpy as np
import pandas as pd
import os 
import pickle
import re
import stanza
import sys
import argparse
import time


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

def load_model(model_name):
    if model_name == 'logreg':
        model = LogisticRegression(solver = 'saga',multi_class='auto', max_iter=10000)
    elif model_name == 'ridge':
        model = RidgeCV()
    elif model_name == "svm":
        model = SVC(kernel = 'linear', C = 1,max_iter=10000)
    return model


def model_training(X,y, mode, model):
        if mode == 'baseline':
            X_train, X_test, y_train, y_test = train_test_split(np.array(X).reshape(-1,1), y, random_state = 42)
        elif mode == 'training':
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

def baseline(embeddings, labels, model):
    print(f'measuring baseline')
    wordcount_ = [x[-1] for x in embeddings]
    audio_len_ = [x[-2] for x in embeddings]
    y = np.array(labels)
    scoring = []
    for i, X in enumerate([wordcount_, audio_len_]):
        lookup = {0: "WC-base", 1: "AL-base"}
        result = [model_training(X,y,'baseline', model), lookup[i]]
        for layer in range(0,12):
            scoring.append([layer]+result)
    return scoring


def iter_layers(embeddings, labels, lay, model):
   
    y = np.array(labels)
    scoring = []
    print(f"running model on layer {lay}")
    # just audio 
    # X = [np.delete(x[lay], [-2,-1]) for x in embeddings]
    X = [np.delete(x, [-2,-1]) for x in embeddings]
    scoring.append([lay, model_training(X, y,'training', model), 'embedding'])
    # audio + audiolen
    X = [np.delete(x, -1) for x in embeddings]
    scoring.append([lay, model_training(X, y,'training', model), 'audiolen'])
    # everything
    X = [x for x in embeddings]
    scoring.append([lay, model_training(X, y,'training', model), 'everything'])
    # audio+wordcount
    X = [np.delete(x, -2) for x in embeddings]
    scoring.append([lay, model_training(X, y,'training', model), 'wordcount'])
    print(scoring)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='controlling number of data and layer used in model training')
    parser.add_argument('--layer',
                    type = int, default = 0, metavar='layer',
                    help = 'add the layer number to train the regression model on'
    )
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
    dataset_dict = {"scc-hubert":'hubert_base_ls960_spokencoco_extracted.pt', 
                    'scc-wav2vec' :'wav2vec_spokencoco_extracted.pt',
                    'libri-wav2vec':'wav2vec_small_librispeech_extracted.pt',
                    'libri-hubert': 'hubert_base_ls960_librispeech_extracted.pt',
                    'fast-vgs': '/home/gshen/work_dir/spoken-model-syntax-probe/fast_vgs_spokencoco_val_extracted.pt'
                    }
    dataset = dataset_dict[args.dataset]
    embeddings, labels, annot, wav = zip(*torch.load(dataset))

    embeddings = [x[args.layer] for x in embeddings][:args.num_data]
    labels = labels[:args.num_data]
    
    if args.baseline == True:
        baseline_score = baseline(embeddings,labels,model=load_model(args.modelname))
        print(f'the baseline score is \n {baseline_score}')
    elif args.model == True:
        iter_layers(embeddings=embeddings, labels = labels, lay = args.layer,model=load_model(args.modelname))

