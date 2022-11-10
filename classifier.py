import torch
import numpy as np
import pandas as pd
import os 
import pickle
import re
import stanza

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

work_path = '/home/gshen/work_dir'
spokencoco_extracted = 'spokencoco_extracted.pt'
libri_train_clean = 'train-clean-100-extracted.pt'


def iter_layers(embeddings, labels, total_layers = 12, audio_len = False, wordcount = False):

    embeddings = embeddings[:5000]
    labels = labels[:5000]
    for lay in range(0,total_layers):
        if not audio_len and not wordcount:
            X = [np.delete(x[lay], [-2,-1]) for x in embeddings]
        elif not audio_len:
            X = [np.delete(x[lay], -2) for x in embeddings]
        elif not wordcount:
            X = [np.delete(x[lay], -1) for x in embeddings]
        else:
            X = [x[lay] for x in embeddings]
        
        y = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
        # print(f"finished splitting data into train and test sets, start training classifiers")
        svm_model_linear = SVC(kernel = 'linear', C = 1,max_iter=10000).fit(X_train, y_train)
        # svm_predictions = svm_model_linear.predict(X_test)
        accuracy = svm_model_linear.score(X_test, y_test)
        # y_pred = svm_model_linear.predict(X_test)
        print(f'the accuracy of the SVC classifier using the embeddings from layer {lay} is {accuracy}')
        # logreg = LogisticRegression(solver = 'saga',multi_class='auto', max_iter=100000)
        # logreg.fit(X_train, y_train)
        # y_pred = logreg.predict(X_test)
        # acc1 = accuracy_score(y_test,y_pred)
        # print(f'the accuracy of the logreg classifier using the embeddings from layer {lay} is {acc1}')

if __name__ == "__main__":
    print(f"start loading embedding for model")
    dataset = spokencoco_extracted
    embeddings, labels = zip(*torch.load(dataset))
    print(f"finished extracting embeddings for model")
    print(f'no audiolen or wordcount')
    iter_layers(embeddings=embeddings, labels = labels)
    print(f'with audio len')
    iter_layers(embeddings=embeddings, labels = labels,audio_len=True)
    print(f'with wordcount')
    iter_layers(embeddings=embeddings, labels = labels,wordcount=True)
    print(f'with both')
    iter_layers(embeddings=embeddings, labels = labels,audio_len=True,wordcount=True)


