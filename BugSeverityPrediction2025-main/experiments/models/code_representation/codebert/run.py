# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import numbers

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer, RobertaModel)

from models.code_representation.codebert.CodeBertModel import CodeBertModel

from models.evaluation.evaluation import evaluate_result, evaluatelog_result

from models.code_representation.codebert.EL_CodeBertLSATModel import EL_CodeBertLSATModel

from models.code_representation.codebert.EL_CodeBertwoAttentionModel import EL_CodeBertwoAttentionModel

from models.code_representation.codebert.EL_CodeBertwoLSTMModel import EL_CodeBertwoLSTMModel
logger = logging.getLogger(__name__)
max_feature_length = []
max_code_length = []


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 label,
                 num_features

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.num_features = num_features


def codebert(js, tokenizer, args):
    num_features = []
    code = ' '.join(js['code_no_comment'].split())
    code_tokens = tokenizer.tokenize(code)
    code_tokens = code_tokens[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    attention_mask = [1] * len(source_ids) + [0] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, attention_mask, js['label'], num_features)


def el_codebert(js, tokenizer, args):
    num_features = []
    code = ' '.join(js['code_no_comment'].split())
    code_tokens = tokenizer.tokenize(code)
    code_tokens = code_tokens[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    attention_mask = [1] * len(source_ids) + [0] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, attention_mask, js['label'], num_features)

def el_codebertwoattention(js, tokenizer, args):
    num_features = []
    code = ' '.join(js['code_no_comment'].split())
    code_tokens = tokenizer.tokenize(code)
    code_tokens = code_tokens[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    attention_mask = [1] * len(source_ids) + [0] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, attention_mask, js['label'], num_features)

def el_codebertwolstm(js, tokenizer, args):
    num_features = []
    code = ' '.join(js['code_no_comment'].split())
    code_tokens = tokenizer.tokenize(code)
    code_tokens = code_tokens[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    attention_mask = [1] * len(source_ids) + [0] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, attention_mask, js['label'], num_features)


# creating features based on args value for different architectures
def convert_examples_to_features(js, tokenizer, args):
    model_arch = args.model_arch
    if model_arch == 'CodeBERT':
        return codebert(js, tokenizer, args)
    elif model_arch == 'EL_CodeBert':
        return el_codebert(js, tokenizer, args)
    elif model_arch == 'EL_CodeBertwoAttention':
        return el_codebertwoattention(js, tokenizer, args)
    elif model_arch == 'EL_CodeBertwoLSTM':
        return el_codebertwolstm(js, tokenizer, args)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("attention_mask: {}".format(' '.join(map(str, example.attention_mask))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].attention_mask), torch.tensor(self.examples[i].label), torch.tensor(
            self.examples[i].num_features)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def train(args, train_dataset, model, tokenizer, lr, epoch, batch_size, fine_tune):
    training_phase = ""
    if not fine_tune:
        training_phase = "Training"
        for param in model.encoder.base_model.parameters():
            param.requires_grad = False
    else:
        training_phase = "Fine Tuning"
        for param in model.encoder.base_model.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total param {}".format(total_params))
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=batch_size, num_workers=4, pin_memory=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)

    # Train!
    logger.info("***** Running {} *****".format(training_phase))
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", epoch)
    logger.info("  batch size = %d", batch_size)
    logger.info("  Total optimization steps = %d", max_steps)
    best_f1 = 0.0
    model.zero_grad()
    losses_eval = []
    losses_train = []

    last_loss = 100
    patience = 5
    trigger_times = 0

    for idx in range(epoch):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []

        for step, batch in enumerate(bar):
            # inputs = batch[0].to(args.device)
            # labels = batch[1].to(args.device)
            # num_features = batch[2].to(args.device)
            #inputs, attention_mask, labels, num_features = [b.to(args.device) for b in batch]
            # print(len(batch))  # 输出: 4
            # print(batch)  # 输出: [input_ids, attention_mask, token_type_ids, labels]

            #inputs, attention_mask, labels = [b.to(args.device) for b in batch]
            inputs = batch[0].to(args.device)
            # label = batch[1].to(args.device)
            # num_features = batch[2].to(args.device)
            attention_mask = batch[1].to(args.device)
            labels = batch[2].to(args.device)
            #print("inputs shape:", inputs.shape)

            model.train()
            #loss, logits = model(input_ids=inputs,attention_mask=attention_mask, num_features=num_features, labels=labels)
            #outputs = model(input_ids=inputs, attention_mask=attention_mask, num_features=num_features, labels=labels)
            loss, logits = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)

            # loss = outputs.loss

            # 检查输出类型并处理不同情况
            # if isinstance(outputs, tuple):
            #     loss = outputs[0]  # 如果输出是元组，第一个元素是loss
            # else:
            #     loss = outputs.loss  # 如果输出是对象，直接获取loss属性
            #
            # #logits = outputs.logits
            #
            # # 检查输出类型并处理不同情况
            # if isinstance(outputs, tuple):
            #     logits = outputs[1]  # 如果输出是元组，第一个元素是loss
            # else:
            #     logits = outputs.logits  # 如果输出是对象，直接获取loss属性

            #logits = torch.softmax(logits, -1)#此时logits等同于prob

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx, round(np.mean(losses), 3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        results = evaluate(args, model, tokenizer)
        log_result(results, logger)
        losses_eval.append(results["eval_loss"])
        losses_train.append(np.mean(losses))

        if not fine_tune:
            if results["eval_loss"] > last_loss:
                trigger_times += 1
                logger.info('  Trigger Times Increased to %d', trigger_times)

                if trigger_times >= patience:
                    logger.info('  Early stopping! Training being finished!')
                    return model
            else:
                logger.info('  Trigger times reset to 0')
                trigger_times = 0

            last_loss = results["eval_loss"]

        # Save model checkpoint
        if results['eval_f1'] > best_f1:
            best_f1 = results['eval_f1']
            logger.info("  " + "*" * 20)
            logger.info("  Best F1:%s", round(best_f1, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            # add arg parameters
            output_dir = os.path.join(output_dir, model_filename(args))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

    return model


def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    all_labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        # label = batch[1].to(args.device)
        # num_features = batch[2].to(args.device)
        attention_mask = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            # lm_loss, logit = model(inputs, label)
            #lm_loss, logit = model(input_ids=inputs, num_features=num_features, labels=label)
            #outputs = model(input_ids=inputs, attention_mask=attention_mask, num_features=num_features, label=label)
            lm_loss, logit = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            #loss = outputs.loss

            #lm_loss = outputs.loss
            # 检查输出类型并处理不同情况
            # if isinstance(outputs, tuple):
            #     lm_loss = outputs[0]  # 如果输出是元组，第一个元素是loss
            # else:
            #     lm_loss = outputs.loss  # 如果输出是对象，直接获取loss属性
            #
            # # logit = outputs.logits
            #
            # # 检查输出类型并处理不同情况
            # if isinstance(outputs, tuple):
            #     logit = outputs[1]  # 如果输出是元组，第一个元素是loss
            # else:
            #     logit = outputs.logits  # 如果输出是对象，直接获取loss属性
            #
            # logit = torch.softmax(logit, -1)  # 此时logit等同于prob

            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    all_labels = np.concatenate(all_labels, 0)
    preds = logits.argmax(-1)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    probs = logits

    result = evaluate_result(all_labels, preds, probs)
    result["eval_loss"] = float(perplexity)
    return result



def test(args, model, tokenizer):
    # 加载测试数据集
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # logger.info(f"***** Running Test for Round {round_no} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    test_loss = 0.0
    nb_test_steps = 0
    model.eval()
    logits = []
    all_labels = []

    # 遍历测试数据并计算损失
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        label = batch[2].to(args.device)

        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs, attention_mask=attention_mask, labels=label)
            logits.append(logit.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            test_loss += lm_loss.mean().item()
        nb_test_steps += 1

    # 将logits和labels合并，并计算预测
    logits = np.concatenate(logits, 0)
    all_labels = np.concatenate(all_labels, 0)
    preds = logits.argmax(-1)
    probs = logits

    # 调用评估函数并返回指标
    result = evaluatelog_result(all_labels, preds, probs, logger)
    #
    f1_weighted, f1_per_class, accuracy, precision, recall, roc_auc, mcc = (
        result[0], result[1], result[2],
        result[3], result[4], result[5], result[6]
    )
    return f1_weighted, f1_per_class, accuracy, precision, recall, roc_auc, mcc


def log_result(result, logger):
    for key, value in sorted(result.items()):
        if isinstance(value, numbers.Number):
            value = round(value, 4)
        logger.info("  %s = %s", key, value)


def model_filename(args):
    # return '{}'.format(
    #     '{}-bs:{}-tb:{}-eb:{}.bin'.format(args.model_arch, args.block_size, args.train_batch_size,
    #                                       args.eval_batch_size))
    return '{}'.format('{}.bin'.format(args.model_arch))



import pandas as pd
pd.options.mode.chained_assignment = None
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
# import tensorflow as tf
# from tensorflow.keras import optimizers, callbacks
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import *
# from tensorflow.keras import mixed_precision
# from tensorflow.keras import backend as K
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
# from format.format_train import FormatData
# from evaluate.evaluation_val import Evaluate
import numpy as np
import random
import pathlib
os.chdir(os.path.dirname(__file__))



class config:
    '''
    configure parameters and paths
    '''
    n_classes = 4
    # TrainValRatio = [0.8, 0.2]
    # TrainInpPath = '../../data/proced/FordA/train_input.csv'
    # TrainOupPath = '../../data/proced/FordA/train_target.csv'
    ParaSavePath = '../../para/parameter.csv'
    ModelSavePath = '../../save_model/FordA/'

    SavePath = '../../para/FordA/test_result.csv'

    # bounds_cost = [(0, 1),(1,20)]



    # 少数类的成本在 [0, 1] 范围内
    # 多数类成本相对于少数类的比率在 [1, 20] 之间
    # 其他类的比率在 [1, 10] 之间
    bounds_cost = [(0, 0.1),  # 少数类成本范围
                   (1, 10),  # 多数类成本比率范围（多数类相对少数类的比率）
                   (1, 3)]  # 其他类的比率范围（其余类）



    # batch, timestep, filter, layer, kernel, dropout, lr
    # bounds_other = [(10, 300),(10, 500),(1, 100),(1, 10),(1, 100),(0,1),(0.001, 0.1)]
    # bounds_all = bounds_cost + bounds_other
    bounds_all = bounds_cost
    F_c = 0.7
    EarlyStopStep = 3

    # maxiter = 10
    # F_list = [(0.8, 1), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2)]
    # k_list = [50, 30, 20, 10, 5, 5, 5, 5, 5, 5]  # length of popsize must be equal to maxiter

    # maxiter = 5
    # F_list = [(0.8,1),(0.4,0.6),(0,0.2),(0,0.2),(0,0.2),]
    # k_list = [ 15, 10, 5, 5, 5]  # length of popsize must be equal to maxiter

    # maxiter = 3
    # F_list = [(0.4,0.6),(0,0.2),(0,0.2),]
    # k_list = [ 10, 5, 5]  # length of popsize must be equal to maxiter


    # maxiter = 3
    # maxiter = 2

    maxiter = 5

    F_list = [(0.4,0.6),(0.2,0.4),(0,0.2),(0,0.2)]
    # k_list = [ 8,6,4,4]  # length of popsize must be equal to maxiter

    # k_list = [4, 4, 4]
    k_list = [10, 8, 6,  4, 4]

    beta=0.5


def SelectTopK(InitialArray,Best_F1,step,k):
    '''
    function to select the top k candidates
    '''
    if step == 0:
        return InitialArray,Best_F1
    else:
        topkidx = sorted(range(len(Best_F1)), key=lambda i: Best_F1[i])[-k:]
        topkidx.sort()
        topkarray = [InitialArray[i] for i in topkidx]
        Best_F1_ = [Best_F1[i] for i in topkidx]

        return topkarray,Best_F1_



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_arch", default="ConcatInline", type=str, required=True,
                        help="model type for training")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=42,
                        help="num_train_epochs")
    parser.add_argument('--num_finetune_epochs', type=int, default=5,
                        help="num_finetune_epochs")
    parser.add_argument("--train_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--finetune_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--finetune_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    if not os.path.exists("log"):
        os.makedirs("log")
    file_handler = logging.FileHandler(os.path.join("log", "log-{}.txt".format(model_filename(args))))
    logger.addHandler(file_handler)

    # Set seed
    set_seed(args.seed)

    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 4
    config.output_hidden_states = True

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    # choosing between different architectures based on the arg value
    #################################
    model_arch = args.model_arch
    if model_arch == 'CodeBERT':
        model = RobertaModel.from_pretrained(args.model_name_or_path, config=config, add_pooling_layer=False)
        model = CodeBertModel(model, config, tokenizer, args)
    elif model_arch == 'EL_CodeBert':
        model = RobertaModel.from_pretrained(args.model_name_or_path, config=config, add_pooling_layer=False)
        model = EL_CodeBertLSATModel(model, config, tokenizer, args)
    elif model_arch == 'EL_CodeBertwoAttention':
        model = RobertaModel.from_pretrained(args.model_name_or_path, config=config, add_pooling_layer=False)
        model = EL_CodeBertwoAttentionModel(model, config, tokenizer, args)
    elif model_arch == 'EL_CodeBertwoLSTM':
        model = RobertaModel.from_pretrained(args.model_name_or_path, config=config, add_pooling_layer=False)
        model = EL_CodeBertwoLSTMModel(model, config, tokenizer, args)
    ##################################

    # multi-gpu training (should be after apex fp16 initialization)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        print("train_dataset size:", len(train_dataset))

        model = train(args, train_dataset, model, tokenizer, lr=args.train_learning_rate, epoch=args.num_train_epochs,
                      batch_size=args.train_batch_size, fine_tune=False)

        model=train(args, train_dataset, model, tokenizer, lr=args.finetune_learning_rate, epoch=args.num_finetune_epochs,
              batch_size=args.finetune_batch_size, fine_tune=True)
        # RunDE(0, tokenizer, args, model)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        output_dir = os.path.join(output_dir, model_filename(args))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        log_result(result, logger)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        output_dir = os.path.join(output_dir, model_filename(args))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer)
        return results


if __name__ == "__main__":
    main()
