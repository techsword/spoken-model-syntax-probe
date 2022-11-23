import torch
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

def generate_bow_embeddings(dataset):
    '''
    Taking ready made (read: extracted) dataset as input and build a Bag-of-words model with sklearn 
    '''
    doc = []
    treedepth = []
    for item in dataset:
        doc.append(item[-2])
        treedepth.append(item[-3])
        doc = list(map(str.lower, doc))

    unique_words = set(' '.join(doc).split())
    print(f'there are {len(unique_words)} unique words')

    index_dict = {}
    for ind, i in enumerate(sorted(unique_words)):
        index_dict[i] = ind
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    count_occurrences = cv.fit_transform(doc)
    return list(count_occurrences.toarray()), treedepth



def model_training(X,y):
    '''
    Basically following the same function layout as in model_run.py.
    Might actually make sense to consolidate this into the other ones
    '''
    model = RidgeCV()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    model.fit(X_train, y_train)
    model_alpha = model.alpha_
    r2score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    return r2score, mse, model_alpha


if __name__ == "__main__":
    dataset_dict = {'spokencoco':"/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings/wav2vec_small_spokencoco_extracted.pt",'librispeech':"/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings/wav2vec_small_librispeech_extracted.pt"}

    for dataset_name in dataset_dict.keys():
        dataset = torch.load(dataset_dict[dataset_name])
        embeddings, labels = generate_bow_embeddings(dataset)
        result = model_training(embeddings, labels)
        print(f"{dataset_name}, {result}, BOW")