import os
import pandas as pd
import torch

# def generate_sentence_embeddings(dataset):
#     from transformers import DebertaTokenizer, DebertaModel
#     import torch
#     import numpy as np

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
#     model = DebertaModel.from_pretrained("microsoft/deberta-base")
#     feat_list = []
#     lab_list = []
#     annot_list = []
#     wav_path_list = []
#     for waveform, annot, depth, path in dataset:
#         lab_list.append(depth)
#         annot_list.append(annot)
#         wav_path_list.append(path)
#         with torch.inference_mode():
#             inputs = tokenizer(annot, return_tensors="pt")
#             outputs = model(**inputs, output_hidden_states = True)
#             features = outputs.hidden_features
#             # features, _ = model.to(device).extract_features(inputs.to(device))
#             # print(features)
#             features = [torch.mean(x.cpu(),dim=1).squeeze().numpy() for x in features]
#             # feat_list.append(torch.mean(features,dim=1).squeeze().numpy())
#             feat_list.append(features)
#     print(f"there are {len(lab_list)} in the extracted dataset, each tensor is {features[0].shape}, the max tree depth is {max(lab_list)} and the min is {min(lab_list)}")
#     return list(zip(feat_list, lab_list,annot_list,wav_path_list))

def generate_sentence_embeddings_(list_of_sents):
    from transformers import DebertaTokenizer, DebertaModel
    import torch
    import numpy as np

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    model = DebertaModel.from_pretrained("microsoft/deberta-base").to(device)
    feat_list = []
    annot_list = []
    for annot in list_of_sents:
        annot_list.append(annot)
        with torch.inference_mode():
            inputs = tokenizer(annot, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states = True)
            features = outputs.hidden_states
            # features, _ = model.to(device).extract_features(inputs.to(device))
            # print(features)
            features = [torch.mean(x.cpu(),dim=1).squeeze().numpy() for x in features]
            # feat_list.append(torch.mean(features,dim=1).squeeze().numpy())
            feat_list.append(features)
    print(f"there are {len(feat_list)} in the extracted dataset, each tensor is {features[0].shape}")
    return list(zip(feat_list, annot_list))

if __name__ == "__main__":
    csv_files = ['/home/gshen/work_dir/spoken-model-syntax-probe/spokencoco_val.csv', '/home/gshen/work_dir/spoken-model-syntax-probe/librispeech_train-clean-100.csv']
    csv_lookup = dict(zip([os.path.basename(x)[:-4].split(sep='-')[0] for x in csv_files],csv_files))

    for x in csv_lookup:
        csv_file = csv_lookup[x]
        save_file = os.path.join('extracted_embeddings','deberta_'+x+'_sentemb.pt')
        # print(save_file)
        if os.path.isfile(save_file):
            print(f"{save_file} exists already! skipping")
        else:
            df = pd.read_csv(csv_file,header=None)
            list_of_sents = list(df.iloc[:,-1])
            extracted_embeddings = generate_sentence_embeddings_(list_of_sents)
            # torch.save(extracted_embeddings, save_file)
            if 'libri' in x:
                tree_depths = [x[1:] for x in torch.load('/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings/wav2vec_small_librispeech_train_extracted.pt')]
            elif 'spokencoco' in x:
                tree_depths = [x[1:] for x in torch.load('/home/gshen/work_dir/spoken-model-syntax-probe/extracted_embeddings/wav2vec_small_spokencoco_val_extracted.pt')]
            sent_embedding = [list(i[0])+list(j) for i,j in zip(extracted_embeddings, tree_depths) if i[1] == j[1]]
            torch.save(sent_embedding, save_file)

