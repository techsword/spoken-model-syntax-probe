import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import plotnine as p9



def make_AL_WC_plot(df_name, num_bin = 8):
    df_dict = {'libri-train':'/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings/hubert_base_ls960_librispeech_train_extracted.pt','scc-val': '/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings/hubert_base_ls960_spokencoco_val_extracted.pt'}
    dataset = torch.load(df_dict[df_name])
    save_path = 'figs/meta/'
    datalist = []
    for sent in dataset:
        datalist.append(list(sent[0][0][-2:])+[sent[1]])

    df = pd.DataFrame(datalist, columns=['AL','WC','TreeDepth'])


    # Plotting wordcount metadata
    min_wc, max_wc = int(min(df['WC'])), int(max(df['WC']))
    WC_bin = list(range(min_wc, max_wc+1, int(np.ceil((max_wc-min_wc)/num_bin))))
    df['WC-bin']=pd.cut(df['WC'],WC_bin)
    wc_plot = (
        p9.ggplot(df.dropna(), p9.aes('WC-bin', fill = 'WC-bin'))
        + p9.geom_bar()
        + p9.ggtitle(df_name.capitalize() + ' Wordcount')
        + p9.xlab('# Sent')
        + p9.ylab('Wordcount by bin')
        + p9.coord_flip()
    )
    wc_plot.save(os.path.join(save_path,'wordcount-'+df_name+'.png'))
    #Plotting audiolength metadata
    min_al, max_al = int(min(df['AL'])), int(max(df['AL']))
    AL_bin = list(range(min_al, max_al, int(np.ceil((max_al-min_al)/num_bin))))
    df['AL-bin']=pd.cut(df['AL'],AL_bin)
    al_plot = (
        p9.ggplot(df.dropna(), p9.aes('AL-bin', fill = 'AL-bin'))
        + p9.geom_bar()
        + p9.ggtitle(df_name.capitalize() + ' Audio Length')
        + p9.xlab('# Sent')
        + p9.ylab('Audio Length by bin')
        + p9.coord_flip()
    )
    al_plot.save(os.path.join(save_path,'audiolen-'+df_name+'.png'))

    # # Plot correlations between TreeDepth and WC/AL
    # plt.figure()
    treedepth_wc_plot = (
        p9.ggplot(df, p9.aes(x='TreeDepth', y = 'WC'))
        + p9.geom_point()
        + p9.geom_smooth()
        + p9.labs(x = 'TreeDepth', y = 'Wordcount', title='Treedepth vs. Wordcount for ' + df_name.capitalize())
    )
    treedepth_wc_plot.save(os.path.join(save_path,'TDvsWC-'+df_name+'.png'))
    treedepth_al_plot = (
        p9.ggplot(df, p9.aes(x='TreeDepth', y = 'AL'))
        + p9.geom_point()
        + p9.geom_smooth()
        + p9.labs(x = 'TreeDepth', y = 'Audio Length', title='Treedepth vs. Audio Length for ' + df_name.capitalize())
    )
    treedepth_al_plot.save(os.path.join(save_path,'TDvsAL-'+df_name+'.png'))

    # return wc_plot, al_plot, treedepth_al_plot, treedepth_wc_plot



def ridge_out_plot(ridge_out_file, overwrite = False):
    results = pd.read_csv(ridge_out_file, header=None, names=['layer', 'r2score', 'mse', 'model_alpha', 'feature'])
    results.replace(r'\[*\]*\s*\'*','', regex=True, inplace=True) 
    results['layer'] = results['layer'].astype('int')
    results['layer'] += 1
    model_layer = max(results['layer'])
    save_path = 'figs/ridge'
    figure_title = os.path.basename(ridge_out_file)[:-4]
    r2_figure_save = os.path.join(save_path,'r2score-'+figure_title+'.png')
    mse_figure_save = os.path.join(save_path,'mse-'+figure_title+'.png')
    if 'libri' in ridge_out_file:
        baseline_file = os.path.join('ridge-results', [x for x in os.listdir('ridge-results') if 'baselines-libri' in x][0])
    elif 'scc' in ridge_out_file:
        baseline_file = os.path.join('ridge-results', [x for x in os.listdir('ridge-results') if 'baselines-scc' in x][0])
    baselines = pd.read_csv(baseline_file, header=None, names=['layer', 'r2score', 'mse', 'model_alpha', 'feature'])
    baselines.replace(r'\[*\]*\s*\'*','', regex=True, inplace=True) 
    baselines['layer'] += 1
    df = pd.concat([results,baselines])
    df.loc[df['feature'] == 'AL', 'feature'] = 'EMB+AL'
    df.loc[df['feature'] == 'WC', 'feature'] = 'EMB+WC'
    # return df
    
    df['layer'] = df['layer'].astype('int')

    df = df[df['layer'] <= int(model_layer)]
    df.sort_values(by=['feature'], inplace=True)
    df = df.reset_index(drop=True)
    df['model_alpha'] = pd.Categorical(df.model_alpha)


    r2_figure = (p9.ggplot(df,p9.aes('layer', 'r2score', color='feature'))
                + p9.geom_point() 
                + p9.geom_line() 
                + p9.theme_linedraw()
                + p9.xlab("Transformer Layer")
                + p9.ylab("R2score")
                + p9.ggtitle(figure_title + ' regression model score')
                + p9.scale_x_continuous(breaks = range(max(df['layer'])+1))
    )
    r2_figure.save(r2_figure_save)
    mse_figure = (p9.ggplot(df,p9.aes('layer', 'mse', color='feature'))
                + p9.geom_point() 
                + p9.geom_line() 
                + p9.theme_linedraw()
                + p9.xlab("Transformer Layer")
                + p9.ylab("Mean Squared Error")
                + p9.ggtitle(figure_title + ' regression model score')
                + p9.scale_x_continuous(breaks = range(max(df['layer'])+1))
    )
    mse_figure.save(mse_figure_save)


def draw_rsa_figs(rsa_out_file, overwrite = False):
    save_path = 'figs/rsa/'
    dataset_lookup = {'libri':'librispeech_train','spoken':'spokencoco_val'}
    for x in dataset_lookup:
        save_file = os.path.join(save_path, os.path.basename(rsa_out_file)[:-4]+'_' + x +'.png')
        if os.path.isfile(save_file) and not overwrite:
            print(f'{save_file} exists already! skipping plotting')
        else:
            df = pd.read_csv(rsa_out_file,names=['model','dataset','layer','alpha','pearsonr', 'p-value'])
            df = df.sort_values(by=['alpha','model','dataset','layer'])
            df.replace(r'\'*\(*\)*\s*','', regex=True, inplace=True) 
            df = df.reset_index(drop=True)
            df[['alpha', 'p-value', 'layer', 'pearsonr']] = df[['alpha', 'p-value', 'layer', 'pearsonr']].apply(pd.to_numeric)
            df = df.groupby(['model','dataset','layer'], as_index=False)['pearsonr'].median()
            df = df[df['dataset'] == dataset_lookup[x]]
            df = df.groupby(['model','layer'], as_index=False)['pearsonr'].mean()
            plot = (p9.ggplot(df, p9.aes('layer', 'pearsonr', color='model'))#, shape = 'dataset'))
                    + p9.geom_point() 
                    + p9.geom_line() 
                    + p9.theme_linedraw()
                    + p9.scale_x_continuous(breaks = range(max(df['layer'])+1))
                    + p9.xlab('Model Transformer Layer')
                    + p9.ylab('Pearson Correlation')
                    + p9.ggtitle('Pearson correlation Score Chart for the different Models'))
            # return plot   
            plot.save(save_file)


if __name__ == "__main__":
    # for dataset in ['libri-train', 'scc-val']:
    #     make_AL_WC_plot(dataset)

    result_path = 'ridge-results/'
    result_files = [os.path.join(result_path,x) for x in os.listdir(result_path) if 'ridge' in x]
    for file in result_files:
        ridge_out_plot(file, overwrite=True)
    
    # draw_rsa_figs('rsa.out')
    # draw_rsa_figs('rsa_delexed.out', True)
    
