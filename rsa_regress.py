import numpy as np
import torch
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import os

from scipy.stats import pearsonr

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
import ursa.util as U
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

def pearson_r_score(Y_true, Y_pred): 
     r =  U.pearson_r(Y_true, Y_pred, axis=0).mean() 
     return r

class Regress:

    default_alphas = [ 10**n for n in range(-3, 2) ]
    metrics = dict(mse       = make_scorer(mean_squared_error, greater_is_better=False),
                   r2        = make_scorer(r2_score, greater_is_better=True),
                   pearson_r = make_scorer(pearson_r_score, greater_is_better=True))
                   
    
    def __init__(self, cv=10, alphas=default_alphas):
        self.cv = cv
        self.grid =  {'alpha': alphas }
        self._model = GridSearchCV(Ridge(), self.grid, scoring=self.metrics, cv=self.cv, return_train_score=False, refit=False)

    def fit(self, X, Y):
        self._model.fit(X, Y)
        result = { name: {} for name in self.metrics.keys() }
        for name, scorer in self.metrics.items():
            mean = self._model.cv_results_["mean_test_{}".format(name)] 
            std  = self._model.cv_results_["std_test_{}".format(name)]
            best = mean.argmax()
            result[name]['mean'] = mean[best] * scorer._sign
            result[name]['std']  = std[best]
            result[name]['alpha'] = self.grid['alpha'][best]
        self._report = result

    def fit_report(self, X, Y):
        self.fit(X, Y)
        return self.report()

    def report(self):
        return self._report


            
def embed(X, ref, sim, parallel=True):
    return U.pairwise(sim, X, ref)



def get_embedding_with_ref(emb, similarity_function = cosine_similarity, seed = 42):
    random.seed(seed)
    ref = random.choices(emb, k = 5)
    return embed(emb, ref, similarity_function)

def main():
    
    X = get_embedding_with_ref()
    Y = get_embedding_with_ref()