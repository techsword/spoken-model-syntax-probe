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
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

work_path = '/home/gshen/work_dir'
spokencoco_extracted = 'spokencoco_extracted.pt'
libri_train_clean = 'train-clean-100-extracted.pt'



num_data = 10000

def logreg_training(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

        # svm_model_linear = SVC(kernel = 'linear', C = 1,max_iter=10000).fit(X_train, y_train)
        # svm_predictions = svm_model_linear.predict(X_test)
        # accuracy = svm_model_linear.score(X_test, y_test)

        logreg = LogisticRegression(solver = 'saga',multi_class='auto', max_iter=10000)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        # acc1 = accuracy_score(y_test,y_pred)
        y_pred = logreg.predict(X_test)
        r2score = r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test,y_pred)
        acc = accuracy_score(y_test,y_pred)
        return r2score, mse, acc

def baseline(embeddings, labels):
    print(f'measuring baseline')
    wordcount_ = [x[-1] for x in embeddings]
    audio_len_ = [x[-2] for x in embeddings]
    y = np.array(labels)
    for i, X in enumerate([wordcount_, audio_len_]):
        lookup = {0: "wordcount", 1: "audiolen"}
        X_train, X_test, y_train, y_test = train_test_split(np.array(X).reshape(-1,1), y, random_state = 42)
        # print(f'the r2_score of the SVC classifier using the embeddings from layer {lay} is {r2score}')
        logreg = LogisticRegression(solver = 'lbfgs',multi_class='auto', max_iter=10000, n_jobs = 6)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        # acc1 = accuracy_score(y_test,y_pred)
        y_pred = logreg.predict(X_test)
        r2score = r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test,y_pred)
        acc = accuracy_score(y_test,y_pred)
        print(f'feature: {lookup[i]}, mse: {mse}, r2score: {r2score}, acc: {acc}')

def iter_layers(embeddings, labels, lay):
   
    y = np.array(labels)
    # for lay in range(0,total_layers):
    scoring = []
    print(f"running logreg on layer {lay}")
    # just audio 
    print(f"logreg on just audio")
    # X = [np.delete(x[lay], [-2,-1]) for x in embeddings]
    X = [np.delete(x, [-2,-1]) for x in embeddings]
    scoring.append([lay, logreg_training(X, y), 'embedding'])
    # audio + audiolen
    print(f"logreg on audio + audiolen")
    X = [np.delete(x, -1) for x in embeddings]
    scoring.append([lay, logreg_training(X, y), 'audiolen'])
    # everything
    print(f"logreg on everything")
    X = [x for x in embeddings]
    scoring.append([lay, logreg_training(X, y), 'everything'])
    # audio+wordcount
    print(f"logreg on audio + wo    rdcount")
    X = [np.delete(x, -2) for x in embeddings]
    scoring.append([lay, logreg_training(X, y), 'wordcount'])
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


    args = parser.parse_args()
    time1 = time.time()
    print(f"start loading embedding for model")
    dataset = os.path.join(work_path,spokencoco_extracted)
    embeddings, labels, annot, wav = zip(*torch.load(dataset))

    embeddings = [x[args.layer] for x in embeddings][:args.num_data]
    labels = labels[:args.num_data]
    # print(args)
    time2 = time.time()
    # print(f"start training logreg model for layer {args.layer} with {'all' if args.num_data == None else args.num_data} data")
    # iter_layers(embeddings=embeddings, labels = labels, lay = args.layer)
    if args.baseline == True:
        baseline(embeddings,labels)
    time3 = time.time()
    print(f"total runtime for logreg is {time3-time2}")
    print(f"loading data takes {time2-time1}")

