import numpy as np
import torch

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

tree_kernel_path = '/home/gshen/work_dir/spoken-model-syntax-probe/tree_kernel'
kernel_files = [os.path.join(tree_kernel_path,x) for x in os.listdir(tree_kernel_path) if '_kernel' in x]
kernel_lookup = dict(zip([os.path.basename(x)[:-10] for x in kernel_files], kernel_files))

pairwise_distance_path = '/home/gshen/work_dir/spoken-model-syntax-probe/pairwise_distances'

pairwise_distance_files = [os.path.join(pairwise_distance_path,x) for x in os.listdir(pairwise_distance_path) if 'full' in x]

def main(alpha = 0.5, seed = 42, mode = 'rsa', delexed = True):
    alpha = str(alpha)
    seed = str(seed)
    for x in pairwise_distance_files:

        # print(x)
        if delexed == True:
            addition = '_delexed'
        else:
            addition = None

        if 'librispeech' in x:
            kernel_file = kernel_lookup['libri_trees_'+alpha+'_'+seed+addition]

        elif 'spokencoco' in x:
            kernel_file = kernel_lookup['scc_trees_'+alpha+'_'+seed+addition]
        
        kernel_set = np.array(torch.load(kernel_file), dtype=object)
        kernel = np.array(kernel_set[:,0])
        kernel_pairs = np.array(kernel_set[:,1:])

        calculated_distances = torch.load(x)

        name_list = str.split(os.path.basename(x),sep='_')[:-3]
        modelname = '_'.join(name_list[:-2])
        datasetname = '_'.join(name_list[-2:])
            
        for layer in range(len(calculated_distances[:-1])):
            # audio_distances = [calculated_distances[layer][x[0],x[1]] for x in kernel_pairs]
            audio_distances = np.array([calculated_distances[layer][x[0],x[1]] for x in kernel_pairs])
            if mode == 'rsa':
                r_score = pearsonr(kernel, audio_distances)
                print((modelname, datasetname, layer, alpha) + r_score)

            elif mode == 'rsa-regress':
                R = Regress()
                score = R.fit_report(X =  kernel.reshape(-1, 1), Y = audio_distances)
                print((modelname, datasetname, layer), score)

def plot(alpha = 0.5, seed = 42, mode = 'rsa'):
    import numpy as np
    import matplotlib.pyplot as plt

    import seaborn as sns
    alpha = str(alpha)
    seed = str(seed)
    for x in pairwise_distance_files:

        # print(x)
        
        if 'librispeech' in x:
            kernel_file = kernel_lookup['libri_trees_'+alpha+'_'+seed]

        elif 'spokencoco' in x:
            kernel_file = kernel_lookup['scc_trees_'+alpha+'_'+seed]
        
        kernel_set = np.array(torch.load(kernel_file), dtype=object)
        kernel = np.array(kernel_set[:,0])
        kernel_pairs = np.array(kernel_set[:,1:])

        calculated_distances = torch.load(x)

        name_list = str.split(os.path.basename(x),sep='_')[:-3]
        modelname = '_'.join(name_list[:-2])
        datasetname = '_'.join(name_list[-2:])
            
        for layer in range(len(calculated_distances[:-1])):
            audio_distances = np.array([calculated_distances[layer][x[0],x[1]] for x in kernel_pairs])
            plt.figure()
            sns.scatterplot(x = audio_distances, y = kernel)
            pltname = str(modelname) + "_"+ str(datasetname)+ "_" + str(layer)
            plt.title(pltname)
            # plt.show()
            plt.savefig(os.path.join('figs/rsa_relations',pltname)+'.png')
            plt.clf()

if __name__ == "__main__":
    main(seed = 42, delexed = True)
    main(seed = 666, delexed = True)
    main(seed = 2022, delexed = True)

    # main(seed = 42, mode = 'rsa-regress')
    # main(seed = 666, mode = 'rsa-regress')
    # main(seed = 2022, mode = 'rsa-regress')
    # plot()