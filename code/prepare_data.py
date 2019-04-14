# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# author: kenjewu

"""
This module is used to parse the raw data and process the training data needed for the model.
"""

import json
import os
import pickle
import re
import itertools

import gluonnlp as nlp
import mxnet as mx
import numpy as np
from mxnet import gluon


def clean_str(string):
    """Tokenization/string cleaning.
    Original from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    Args:
        string (str): the input string

    Returns:
        str: processed string
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", r" \( ", string)
    string = re.sub(r"\)", r" \) ", string)
    string = re.sub(r"\?", r" \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def pad_sequences(sequences, max_len, pad_value):
    '''
    Fill the sequence to the specified length, long truncation
    Args:
        sequences: A list of all sentences, a list of list
        max_len: Specified maximum length
        pad_value: Specified fill value
    Returns:
        pades_seqs: A numpy array
    '''

    paded_seqs = np.zeros((len(sequences), max_len))
    for idx, seq in enumerate(sequences):
        paded = None
        if len(seq) < max_len:
            paded = np.array((seq + [pad_value] * (max_len - len(seq))))
        else:
            paded = np.array(seq[0:max_len])
        paded_seqs[idx] = paded

    return paded_seqs


def set_embedding_for_vocab(vocab, wv_name):
    if wv_name == 'glove':
        pretrain_embedding = nlp.embedding.GloVe(
            source='glove.6B.50d', embedding_root='..data/embedding')
    elif wv_name == 'w2v':
        pretrain_embedding = nlp.embedding.Word2Vec(
            source='GoogleNews-vectors-negative300', embedding_root='..data/embedding')
    elif wv_name == 'fasttext':
        pretrain_embedding = nlp.embedding.FastText(
            source='wiki.simple', embedding_root='..data/embedding')
    else:
        pretrain_embedding = None

    if pretrain_embedding is not None:
        vocab.set_embedding(pretrain_embedding)


def preprocess_data(sentences, wv_name, max_seq_len=100):
    clipper = nlp.data.ClipSequence(max_seq_len)
    tokenizer = nlp.data.SpacyTokenizer(lang='en')
    sentences = [clipper(tokenizer(clean_str(sentence))) for sentence in sentences]

    # count tokens and build vocab
    tokens = list(itertools.chain.from_iterable(sentences))
    token_counter = nlp.data.count_tokens(tokens)
    vocab = nlp.Vocab(token_counter)

    # loading pretrained embedding
    set_embedding_for_vocab(vocab, wv_name)

    # Convert all words of sentences their corresponding index in the vocabulary
    sentences_idx = [vocab[sentence] for sentence in sentences]

    return vocab, sentences_idx


def get_data(data_json_path=None, wv_name=None, formated_data_path=None, max_seq_len=100):
    """Process raw data and obtain standard data that can be used for model training.

    Args:   
        data_json_path (str, optional): Defaults to None. 
         the path of raw data. This is a json file.
        wv_name (str, optional): Defaults to None. one of {'glove', 'w2v', 'fasttext', 'random'}.
        formated_data_path (str, optional): Defaults to None.
         The path to save the processed standard data.
        max_seq_len (int, optional): Defaults to 10. max length of every sample.

    Returns:
        dict: dict of data
    """

    if os.path.exists(formated_data_path):
        with open(formated_data_path, 'rb') as f:
            formated_data = pickle.load(f)
    else:
        with open(data_json_path, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
        sentences, labels = data['texts'], data['labels']

        # data processing
        vocab, x = preprocess_data(sentences, wv_name, max_seq_len=max_seq_len)
        y = np.array(labels) - 1

        # save processed data
        formated_data = {'x': x, 'y': y, 'vocab': vocab}
        with open(formated_data_path, 'wb') as fw:
            pickle.dump(formated_data, fw)

    return formated_data


def get_dataloader(dataset, batch_size, is_train=True):

    # Construct the DataLoader Pad data, stack label and lengths
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0),
        nlp.data.batchify.Stack())

    dataloader = None

    # dataloader for training
    if is_train:
        data_lengths = [len(sample[0]) for sample in dataset]

        # n this example, we use a FixedBucketSampler,
        # which assigns each data sample to a fixed bucket based on its length.
        batch_sampler = nlp.data.sampler.FixedBucketSampler(
            data_lengths,
            batch_size=batch_size,
            num_buckets=10,
            ratio=0.2,
            shuffle=True)
        dataloader = gluon.data.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            batchify_fn=batchify_fn)
    # dataloader for not training
    else:
        dataloader = gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            batchify_fn=batchify_fn)

    return dataloader
