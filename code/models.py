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

"""The module includes an attention layer, a two-way LSTM combined
 with a attention mechanism for the model of sentiment classification.
"""


import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn, rnn


class SelfAttention(nn.HybridBlock):
    """An attention layer, cite:
    https://arxiv.org/pdf/1703.03130.pdf

    Args:
        natt_units (int): number of attention hidden units
        natt_hops (int): number of channels with different attention

    Returns:
        NDArray: representation of input after attach attetion
        NDArray: attention value
    """

    def __init__(self, natt_units, natt_hops, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.ut_dense = nn.Dense(
                natt_units, activation='tanh', flatten=False)
            self.et_dense = nn.Dense(natt_hops, activation=None, flatten=False)

    def hybrid_forward(self, F, x):
        # x shape: [batch_size, seq_len, embedding_width]
        # ut shape: [batch_size, seq_len, att_unit]
        ut = self.ut_dense(x)
        # et shape: [batch_size, seq_len, att_hops]
        et = self.et_dense(ut)

        # att shape: [batch_size,  att_hops, seq_len]
        att = F.softmax(F.transpose(et, axes=(0, 2, 1)), axis=-1)
        # output shape [batch_size, att_hops, embedding_width]
        output = F.batch_dot(att, x)

        return output, att


class SelfAttentiveBiLSTM(nn.HybridBlock):
    """A BiLSTM + Attention model for classification task.
    cite: https://arxiv.org/pdf/1703.03130.pdf.

    Args:
        nwords (int): number of words in vocab
        nword_dims (int): dimension of word vector
        nhiddens (int): number of hidden units in lstm cell
        nlayers (int): number of lstm layers
        natt_units (int): number of attention hidden units
        natt_hops (int): number of channels with different attention
        nfc (int): number of hidden units in dense layer
        nclass (int): number of categories
        drop_prob (float): dropout prob
        pool_way (str): method of processing attention coding results, flatten, mean or prune
        nprune_p (str): number of hidden units in the clipping layer
        nprune_q (str): number of hidden units in the clipping layer

    Raises:
        ValueError: 'The attention output is not processed correctly.'
                    ' Please select from a, b, c'
    Returns:
        NDArray: output of model
        NDArray: attention value
    """

    def __init__(self, nwords, nword_dims, nhiddens, nlayers, natt_units, natt_hops, nfc, nclass,
                 drop_prob, pool_way, nprune_p=None, nprune_q=None, **kwargs):
        super(SelfAttentiveBiLSTM, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding_layer = nn.Embedding(nwords, nword_dims)
            self.bilstm = rnn.LSTM(
                nhiddens, num_layers=nlayers, dropout=drop_prob, bidirectional=True)
            self.att_encoder = SelfAttention(natt_units, natt_hops)
            self.dense = nn.Dense(nfc, activation='tanh')
            self.output_layer = nn.Dense(nclass)

            self.dense_p, self.dense_q = None, None
            if all([nprune_p, nprune_q]):
                self.dense_p = nn.Dense(
                    nprune_p, activation='tanh', flatten=False)
                self.dense_q = nn.Dense(
                    nprune_q, activation='tanh', flatten=False)

            self.drop_prob = drop_prob
            self.pool_way = pool_way

    def hybrid_forward(self, F, input):
        # input_embed: [batch, seq_len, nword_dims]
        input_embed = self.embedding_layer(input)
        lstm_output = self.bilstm(F.transpose(input_embed, axes=(1, 0, 2)))
        # att_output: [batch, natt_hops, nword_dims]
        att_output, att = self.att_encoder(F.transpose(lstm_output, axes=(1, 0, 2)))

        if self.pool_way == 'flatten':
            dense_input = F.Dropout(F.flatten(att_output), self.drop_prob)
        elif self.pool_way == 'mean':
            dense_input = F.Dropout(F.mean(att_output, axis=1), self.drop_prob)
        elif self.pool_way == 'prune' and all([self.dense_p, self.dense_q]):
            # p_section: [batch, att_hops, nprune_p]
            p_section = self.dense_p(att_output)
            # q_section: [batch, emsize, nprune_q]
            q_section = self.dense_q(F.transpose(att_output, axes=(0, 2, 1)))
            dense_input = F.Dropout(
                F.concat(F.flatten(p_section), F.flatten(q_section), dim=-1), self.drop_prob)
        else:
            raise ValueError('The attention output is not processed correctly.'
                             ' Please select from a, b, c')

        dense_out = self.dense(dense_input)
        output = self.output_layer(F.Dropout(dense_out, self.drop_prob))

        return output, att
