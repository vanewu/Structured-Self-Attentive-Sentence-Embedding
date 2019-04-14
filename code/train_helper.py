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

# author:kenjewu

"""Function used for auxiliary training
"""


import os
from time import time

import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, nd
from sklearn.metrics import accuracy_score, f1_score


def calculate_loss(x, y, model, loss, loss_name, class_weight, penalization_coeff):
    """calculate loss value

    Args:
        x (NDArray): intput of model
        y (NDArray): target
        model (Block): model
        loss (gluon.loss): loss function
        loss_name (str): name of loss function
        class_weight (NDArray): weight of sample loss value for each category
        penalization_coeff (float): Attention penalty coefficient

    Returns:
        NDArray: output of model
        NDArray: loss value
    """

    pred, att = model(x)
    if loss_name == 'sce':
        l = loss(pred, y)
    elif loss_name == 'wsce':
        l = loss(pred, y, class_weight, class_weight.shape[0])

    # penalty
    diversity_penalty = nd.batch_dot(att, nd.transpose(att, axes=(0, 2, 1))
                                     ) - nd.eye(att.shape[1], ctx=att.context)
    l = l + penalization_coeff * diversity_penalty.norm(axis=(1, 2))

    return pred, l


def train(train_dataloader, valid_dataloader, model, loss, trainer, ctx,
          nepochs, penalization_coeff, clip, class_weight, loss_name,
          model_root, model_name, log_interval, lr_decay_step, lr_decay_rate):
    """Function used in training

    Args:
        train_dataloader (Dataloader): data loader of train data set
        valid_dataloader (Dataloader): data loader of valid data set
        model (Block): model
        loss (gluon.loss): loss function
        trainer (gluon.Trainer): trainer
        ctx (context): mx.gpu() or mx.cpu()
        nepochs (int): number of epoch
        penalization_coeff (float): attention penalty coefficient
        clip (float): value of gradient clip
        class_weight (NDArray): weight of sample loss value for each category
        loss_name (str): name of loss function
        model_root (str): root path of model
        model_name (str): name of model
        log_interval (int): interval steps of log output
        lr_decay_step (int): step of learning rate decay
        lr_decay_rate (float): rate of learning rate decay
    """

    print('Train on ', ctx)

    parameters = [p for p in model.collect_params().values() if p.grad_req != 'null']
    init_lr = trainer.learning_rate
    for epoch in range(1, nepochs + 1):
        start = time()
        best_F1_valid = 0.
        train_loss = 0.
        total_pred = []
        total_true = []

        for nbatch, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x, batch_y = batch_x.as_in_context(ctx), batch_y.as_in_context(ctx)
            with autograd.record():
                batch_pred, l = calculate_loss(batch_x, batch_y, model, loss,
                                               loss_name, class_weight, penalization_coeff)
            l.backward()

            # clip gradient
            if clip is not None:
                grads = [p.grad(ctx) for p in parameters]
                gluon.utils.clip_global_norm(grads, clip)

            # update trainable params (更新参数)
            trainer.step(batch_x.shape[0])

            batch_pred = np.argmax(nd.softmax(batch_pred, axis=1).asnumpy(), axis=1)
            batch_true = np.reshape(batch_y.asnumpy(), (-1, ))
            total_pred.extend(batch_pred.tolist())
            total_true.extend(batch_true.tolist())
            batch_train_loss = l.mean().asscalar()

            train_loss += batch_train_loss

            if (nbatch + 1) % log_interval == 0:
                print('epoch %d, batch %d, bach_train_loss %.4f, batch_train_acc %.3f' %
                      (epoch, nbatch + 1, batch_train_loss, accuracy_score(batch_true, batch_pred)))

        # metric on total train data set
        F1_train = f1_score(np.array(total_true), np.array(total_pred), average='weighted')
        acc_train = accuracy_score(np.array(total_true), np.array(total_pred))
        train_loss /= (nbatch + 1)

        # metrics on valid data set
        F1_valid, acc_valid, valid_loss = evaluate(valid_dataloader, model, loss, loss_name,
                                                   penalization_coeff, class_weight, ctx)

        print('epoch % d, learning_rate % .5f \n\t train_loss % .4f,'
              ' acc_train %.3f, F1_train %.3f, ' %
              (epoch, trainer.learning_rate, train_loss, acc_train, F1_train))
        print('\t valid_loss %.4f, acc_valid %.3f, F1_valid %.3f, '
              '\ntime %.1f sec' % (valid_loss, acc_valid, F1_valid, time() - start))

        # learning rate decay
        if epoch % lr_decay_step == 0:
            trainer.set_learning_rate(init_lr / (1.0 + lr_decay_rate * epoch))

        # save best model structure and parameters
        if F1_valid > best_F1_valid:
            best_F1_valid = F1_valid
            best_epoch = epoch

            model_dir = model_root + os.path.sep + model_name
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, 'self_att_bilstm_model')
            model.export(model_path)
            print('best f1: %d, best epoch: %d, model structure'
                  ' and parameters saved in: %s' % (best_F1_valid, best_epoch, model_path))
        print('=' * 50)


def evaluate(valid_dataloader, model, loss, loss_name, penalization_coeff, class_weight, ctx):
    """the evaluation function

    Args:
        valid_dataloader (Dataloader): data loader of valid data set
        model (Block): model
        loss (gluon.loss): loss function
        loss_name (str): name of loss function
        penalization_coeff (float): attention penalty coefficient
        class_weight (NDArray): weight of sample loss value for each category
        ctx (context): mx.gpu() or mx.cpu()

    Returns:
        float: f1 score of data
        float: accuracy score of data
        float: loss value of data
    """

    valid_loss = 0.
    total_pred = []
    total_true = []

    for nbatch, (batch_x, batch_y) in enumerate(valid_dataloader):
        batch_x, batch_y = batch_x.as_in_context(ctx), batch_y.as_in_context(ctx)
        batch_pred, l = calculate_loss(batch_x, batch_y, model, loss,
                                       loss_name, class_weight, penalization_coeff)
        total_pred.extend(np.argmax(nd.softmax(
            batch_pred, axis=1).asnumpy(), axis=1).tolist())
        total_true.extend(np.reshape(batch_y.asnumpy(), (-1,)).tolist())
        nbatch += 1
        valid_loss += l.mean().asscalar()

    F1_valid = f1_score(np.array(total_true), np.array(total_pred), average='weighted')
    acc_valid = accuracy_score(np.array(total_true), np.array(total_pred))
    valid_loss /= nbatch

    return F1_valid, acc_valid, valid_loss
