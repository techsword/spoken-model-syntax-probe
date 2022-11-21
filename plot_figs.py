import torch
import torchaudio
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser


df_dict = {'libri-train':'/home/gshen/work_dir/spoken-model-syntax-probe/hubert_base_ls960_librispeech_extracted.pt','scc-val': '/home/gshen/work_dir/spoken-model-syntax-probe/hubert_base_ls960_spokencoco_extracted.pt'}


# def return_meta_pd(dataset):
#     datalist = []
#     for sent in dataset:
#         datalist.append(list(sent[0][0][-2:])+[sent[1]])

#     df = pd.DataFrame(datalist, columns=['AL','WC','TreeDepth'])
#     return df

def get_meta_figs(df):
    plt.figure()
    sns.lmplot(data = df, x='TreeDepth', y = 'WC')
    plt.figure()
    sns.regplot(data = df, x='TreeDepth', y = 'AL')

def make_AL_WC_plot(df_name, num_bin = 8):
    dataset = torch.load(df_dict[df_name])
    datalist = []
    for sent in dataset:
        datalist.append(list(sent[0][0][-2:])+[sent[1]])

    df = pd.DataFrame(datalist, columns=['AL','WC','TreeDepth'])

    # Plotting wordcount metadata
    plt.figure()
    min_wc, max_wc = int(min(df['WC'])), int(max(df['WC']))
    WC_bin = list(range(min_wc, max_wc+1, int(np.ceil((max_wc-min_wc)/num_bin))))
    df['WC-bin']=pd.cut(df['WC'],WC_bin)
    wc_plot = sns.countplot(y = df['WC-bin'])
    wc_plot.set(xlabel = '# Sent', ylabel = 'Wordcount Bin')
    plt.savefig("figs/meta/wordcount-"+df_name+".png")

    # Plotting audiolength metadata
    plt.figure()
    min_al, max_al = int(min(df['AL'])), int(max(df['AL']))
    AL_bin = list(range(min_al, max_al, int(np.ceil((max_al-min_al)/num_bin))))
    df['AL-bin']=pd.cut(df['AL'],AL_bin)
    al_plot = sns.countplot(y = df['AL-bin'])
    al_plot.set(xlabel = '# Sent', ylabel = 'Audio length Bin')
    plt.savefig("figs/meta/audiolen-"+df_name+".png")


if __name__ == "__main__":
    for dataset in ['libri-train', 'scc-val']:
        make_AL_WC_plot(dataset)