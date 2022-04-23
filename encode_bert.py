"""
Usage:
  encode_bert_states.py [--input_file=INPUT_FILE] [--output_dir=OUTPUT_DIR] [--split=SPLIT]

Options:
  -h --help                     show this help message and exit
  --input_file=INPUT_FILE       input dir file
  --output_dir=OUTPUT_DIR       write down output file
  --split=SPLIT                 split name

Encoding text with Bert with two methods: average of all words,
 and the cls token as sentence representation.
"""

import numpy as np
from docopt import docopt
import torch
from transformers import *
import pickle
from tqdm import tqdm
import pandas as pd
import argparse
from pathlib import Path
import itertools

#Mean Pooling - Take attention mask into account for correct averaging 
#from https://www.sbert.net/examples/applications/computing-embeddings/README.html
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def read_data_file(input_file):
    """
    read the data file with a pickle format
    :param input_file: input path, string
    :return: the file's content
    """
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data

def read_data(input_file):
    with open(input_file) as f:
        data =f.readlines()
    return data

def read_csv_file(csv_path):
    df = pd.read_csv(csv_path, header=0)
    df.dropna(inplace=True)
    df.to_csv(csv_path, index=False)
    return df['text'].tolist()

#Load AutoModel from huggingface model repository
def load_lm():
    """
    load bert's language model -> load sentence-transformers 
    :return: the model and its corresponding tokenizer
    """
    # model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
    # tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    # model = model_class.from_pretrained(pretrained_weights)
    # return model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    return model, tokenizer

#Tokenize sentences
def tokenize(tokenizer, sentences, num_egs):
    """
    Iterate over the data and tokenize it. Sequences longer than 512 tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: data to tokenize
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    print("tokenizing..")
    if num_egs == -1:
        sentences = sentences
    else:
        sentences = sentences[:num_egs]
    
    # for row in tqdm(data):
    #     tokens = tokenizer.encode(row, add_special_tokens=True) #['hard_text']
    #     # keeping a maximum length of bert tokens: 512
    #     tokenized_data.append(tokens[:512])
    
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
    return encoded_input


def batch_generator(iterable, batch_size=1):
    iterable = iter(iterable)

    while True:
        batch = list(itertools.islice(iterable, batch_size))
        if len(batch) > 0:
            yield batch
        else:
            break

def encode_text(model, data):
     """
     encode the text
     :param model: encoding model
     :param data: data
     :return: two numpy matrices of the data:
                 first: average of all tokens in each sentence
                 second: cls token of each sentence
     """
     all_data_cls = []
     all_data_avg = []
     batch = []
     print("encoding text..")
     for row in tqdm(data):
         batch.append(row)
         input_ids = torch.tensor(batch)
         with torch.no_grad():
             last_hidden_states = model(input_ids)[0]
             all_data_avg.append(last_hidden_states.squeeze(0).mean(dim=0).numpy())
             all_data_cls.append(last_hidden_states.squeeze(0)[0].numpy())
         batch = []
     return np.array(all_data_avg), np.array(all_data_cls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="mono/txt_split/en_de/en.txt")
    parser.add_argument("--output_dir", default="mono/bert_states/en_de/")
    parser.add_argument("--split", default="en")
    parser.add_argument("--num", default=-1, type=int, help="num of examples to encode from the dataset")
    args = parser.parse_args()

    #arguments = docopt(__doc__)

    in_file = args.input_file #arguments['--input_file']
    out_dir = args.output_dir #arguments['--output_dir']
    split = args.split #arguments['--split']
    num_egs = args.num #arguments['--num'] # -1 to encode all sentences in the dataset


    model, tokenizer = load_lm()

    all_data = read_data(in_file)
    print(len(all_data))


    # tokenize sentences
    all_se = []

    for data in batch_generator(all_data, batch_size=100):
        encoded_input = tokenize(tokenizer, data, num_egs)

        #Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        #print(type(sentence_embeddings))
        sentence_embeddings = sentence_embeddings.detach().to('cpu').numpy()
        all_se.extend(sentence_embeddings)

    all_se = np.asarray(all_se)
    print(all_se.shape)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    np.save(out_dir + split + '_se.npy', all_se)

    # avg_data, cls_data = encode_text(model, tokens)

    # np.save(out_dir + '/' + split + '_avg.npy', avg_data)
    # np.save(out_dir + '/' + split + '_cls.npy', cls_data)
    # print(f"encoding done for {split}")

