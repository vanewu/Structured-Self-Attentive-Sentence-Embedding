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

"""This is the training script
"""


import argparse

import gluonnlp as nlp
import mxnet as mx
from mxnet import nd, gluon, init
from mxnet.gluon.data import DataLoader


import train_helper as th
from weighted_softmaxCE import WeightedSoftmaxCE
from models import SelfAttentiveBiLSTM
from prepare_data import get_data, get_dataloader


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu().

    Returns:
        context: mx.gpu() or mx.cpu()
    """

    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def build_dataloader(args):
    """build data loader for model training and validding

    Args:
        args (argparse.Namespace): Command parameter parser

    Returns:
        DataLoader: train dataloader
        DataLoader: valid dataloader
        Vocab: vocab of corpus
    """

    max_seq_len = args.max_seq_len
    valid_rate = args.valid_rate
    batch_size = args.batch_size

    # Get training data and validation data set
    print('Getting the data...')
    data = get_data(args.data_json_path, args.wv_name, args.formated_data_path, max_seq_len)
    x, y, vocab = data['x'], data['y'], data['vocab']

    # get dataloader
    dataset = gluon.data.SimpleDataset([[field1, field2] for field1, field2 in zip(x, y)])
    train_dataset, valid_dataset = nlp.data.train_valid_split(dataset, valid_rate)

    train_dataloader = get_dataloader(train_dataset, batch_size, is_train=True)
    valid_dataloader = get_dataloader(valid_dataset, batch_size, is_train=False)

    return train_dataloader, valid_dataloader, vocab


def build_model(vocab, args):
    """build model denpend on arguments

    Args:
        vocab (Vocab): vocab of corpus
        args (argparse.Namespace): Command parameter parser

    Returns:
        Block: model
    """

    nword_dims = args.nword_dims
    nhiddens = args.nhiddens
    nlayers = args.nlayers
    natt_units = args.natt_units
    natt_hops = args.natt_hops
    nfc = args.nfc
    nclass = args.nclass
    pool_way = args.pool_way
    nprune_p = args.nprune_p
    nprune_q = args.nprune_q
    drop_prob = args.drop_prob
    freeze_embedding = args.freeze_embedding

    # configuration model
    nwords = len(vocab)
    model = SelfAttentiveBiLSTM(nwords, nword_dims, nhiddens, nlayers,
                                natt_units, natt_hops, nfc, nclass,
                                drop_prob, pool_way, nprune_p=nprune_p, nprune_q=nprune_q)
    model.initialize(init=init.Xavier(), ctx=ctx)
    model.hybridize()
    # freeze embedding layer
    if freeze_embedding:
        model.embedding_layer.weight.set_data(vocab.embedding.idx_to_token)
        model.embedding_layer.collect_params().setattr('grad_req', 'null')

    return model


def build_loss_optimizer(model, args, ctx):
    """build loss function and optimizer for training

    Args:
        model (Block): model
        args ([type]): [description]
        ctx (Context): mx.gpu() or mx.cpu()

    Returns:
        gluon.loss: loss function
        gluon.Trainer: trainer
        NDArray: weight of sample loss value for each category
    """

    lr = args.lr
    loss_name = args.loss_name
    optimizer = args.optimizer
    trainer = gluon.Trainer(model.collect_params(),
                            optimizer, {'learning_rate': lr})

    class_weight = None
    if loss_name == 'sce':
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
    elif loss_name == 'wsce':
        loss = WeightedSoftmaxCE()
        # This value is obtained by counting the data samples in advance.
        class_weight = nd.array([3.0, 5.3, 4.0, 2.0, 1.0], ctx=ctx)

    return loss, trainer, class_weight


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nword_dims', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhiddens', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers in BiLSTM')
    parser.add_argument('--natt_units', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--natt_hops', type=int, default=1,
                        help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--nfc', type=int, default=512,
                        help='hidden (fully connected) layer size for classifier MLP')
    parser.add_argument('--pool_way', type=str, choice=['flatten', 'mean', 'prune'],
                        default='flatten', help='pool att output way')
    parser.add_argument('--nprune_p', type=int, default=None, help='prune p size')
    parser.add_argument('--nprune_q', type=int, default=None, help='prune q size')
    parser.add_argument('--nclass', type=int, default=5, help='number of classes')
    parser.add_argument('--wv_name', type=str, choices={'glove', 'w2v', 'fasttext', 'random'},
                        default='random', help='word embedding way')

    parser.add_argument('--drop_prob', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--lr', type=float, default=.001, help='initial learning rate')
    parser.add_argument('--nepochs', type=int, default=10, help='upper epoch limit')
    parser.add_argument('--loss_name', type=str, choice=['sce, wsce'],
                        default='sce', help='loss function name')
    parser.add_argument('--freeze_embedding', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=2018, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--penalization_coeff', type=float, default=0.1,
                        help='the penalization coefficient')

    parser.add_argument('--valid_rate', type=float, default=0.1,
                        help='proportion of validation set samples')
    parser.add_argument('--max_seq_len', type=int, default=100,
                        help='max length of every sample')
    parser.add_argument('--model_root', type=str, default='../models',
                        help='path to save the final model')
    parser.add_argument('--model_name', type=str, default='self_att_bilstm_model',
                        help='path to save the final model')
    parser.add_argument('--data_json_path', type=str,
                        default='../data/sub_review_labels.json', help='raw data path')
    parser.add_argument('--formated_data_path', type=str,
                        default='../data/formated_data.pkl', help='formated data path')

    args = parser.parse_args()

    # Set mxnet random number seed
    mx.random.seed(args.seed)

    # set the useful of gpu or cpu
    ctx = try_gpu()

    # get train and valid dataloader
    train_dataloader, valid_dataloader, vocab = build_dataloader(args)

    # build model
    model = build_model(vocab, args)

    # build loss, trainer and class_weight
    loss, trainer, class_weight = build_loss_optimizer(model, args, ctx)

    # train
    nepochs = args.nepochs
    penalization_coeff = args.penalization_coeff
    clip = args.clip
    loss_name = args.loss_name
    model_root = args.model_root
    model_name = args.model_name
    th.train(train_dataloader, valid_dataloader, model, loss, trainer, ctx,
             nepochs, penalization_coeff, clip, class_weight, loss_name,
             model_name, model_root)
