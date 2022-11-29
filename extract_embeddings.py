import torch
from torchsummary import summary
import os 
import argparse


from custom_classes import Corpus
from custom_functions import walk_librispeech_dirs, read_json_save_csv, generating_features

# model_file = '~/work_dir/spoken-model-syntax-probe/hubert_base_ls960.pt' #'~/work_dir/wav2vec_small.pt' # 'hubert_base_ls960.pt' or 'wav2vec_small.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sr = 16000


# librispeech settings
librispeech_root = '/home/gshen/work_dir/librispeech-train/'# 'LibriSpeech/
libri_split = 'train-clean-100' # 'test-clean'
saved_file = libri_split+'-extracted.pt'



# spokencoco settings

json_path = '/home/gshen/SpokenCOCO/SpokenCOCO_val.json'
root_dir='/home/gshen/SpokenCOCO/'

csv_file = 'spokencoco_val.csv'



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='control what model and which dataset you are using for embedding extraction')

    parser.add_argument('--model',
                    type=str, default = 'hubert', metavar='model',
                    help="choose the model used to extract embeddings, default is hubert. options: hubert, wav2vec, random"
    )
    parser.add_argument('--corpus',
                    type=str, default = 'spokencoco', metavar='corpus',
                    help="choose the corpus to extract embeddings from, default is spokencoco. options: spokencoco, librispeech"
    )
    args = parser.parse_args()

    # setting the model to extract embeddings and the output filename
    model_dict = {'hubert': '/home/gshen/work_dir/spoken-model-syntax-probe/hubert_base_ls960.pt', 'wav2vec':'/home/gshen/work_dir/wav2vec_small.pt'}
    if args.model != 'random':
        model_file = model_dict[args.model]
        saved_file = os.path.basename(model_file[:-3]) + '_' + args.corpus + '_extracted' + '.pt'
        from custom_functions import loading_fairseq_model
        model = loading_fairseq_model(model_file)
    else:
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
        configuration = Wav2Vec2Config()

        # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
        hf_model = Wav2Vec2Model(configuration)

        # Accessing the model configuration
        configuration = hf_model.config

        saved_file = 'wav2vec_random_' + args.corpus + '_extracted.pt'
        from custom_functions import loading_huggingface_model
        model = loading_huggingface_model(hf_model)


    save_path = 'extracted_embeddings'

    if args.corpus == 'spokencoco':
        # extract embeddings with the model defined above from spokencoco corpus
        if os.path.isfile(csv_file) == False:
            # making sure the csv file is there, otherwise create the csv file
            print(f"{csv_file} not found, creating from {json_path}")
            spokencoco_df = read_json_save_csv(json_path)
            spokencoco_df.to_csv(csv_file, header=None, index = None)
        print(f"generating features")
        spokencoco = Corpus(csv_file, root_dir = root_dir)
        spokencoco_extracted = generating_features(spokencoco, model)
        print(f"saving the extracted embeddings to {saved_file}")
        torch.save(spokencoco_extracted, os.path.join(save_path, saved_file))

    elif args.corpus == "librispeech":
        # extract embeddings with the model defined above from librispeech corpus
        if os.path.isfile('librispeech_'+libri_split+'.csv') == False:
            # making sure the csv file is there, otherwise create the csv file
            print(f"{'librispeech_'+libri_split+'.csv'} not found, creating from {os.path.join(librispeech_root, libri_split)}")
            librispeech_dataset_df = walk_librispeech_dirs(librispeech_root=librispeech_root, libri_split=libri_split)
            librispeech_dataset_df.to_csv('librispeech_'+libri_split+'.csv', index=False)
        print(f"generating features")
        libri_ds = Corpus('librispeech_'+libri_split+'.csv', os.path.join(librispeech_root, libri_split))
        libri_extracted = generating_features(libri_ds, model)
        print(f"saving the extracted embeddings to {saved_file}")
        torch.save(libri_extracted, os.path.join(save_path,saved_file))


    
