#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import sys
import os
import argparse
import random
import numpy as np
import torch
from ptranking.utils.args.argsUtil import ArgsUtil

from ptranking.ltr_adhoc.eval.ltr import LTREvaluator, LTR_ADHOC_MODEL
from ptranking.ltr_tree.eval.ltr_tree import TreeLTREvaluator, LTR_TREE_MODEL
from ptranking.ltr_adversarial.eval.ltr_adversarial import AdLTREvaluator, LTR_ADVERSARIAL_MODEL

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
"""
The command line usage:

(1) Without using GPU
python pt_ranking.py -model ListMLE -dir_json /home/dl-box/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/

(2) Using GPU
python pt_ranking.py -cuda 0 -model ListMLE -dir_json /home/dl-box/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/

"""

if __name__ == '__main__':

    """
    >>> Learning-to-Rank Models <<<
    (1) Optimization based on Empirical Risk Minimization
    -----------------------------------------------------------------------------------------
    | Pointwise | RankMSE                                                                   |
    -----------------------------------------------------------------------------------------
    | Pairwise  | RankNet                                                                   |
    -----------------------------------------------------------------------------------------
    | Listwise  | LambdaRank % ListNet % ListMLE % RankCosine %  ApproxNDCG %  WassRank     |
    |           | STListNet  % LambdaLoss                                                   |
    -----------------------------------------------------------------------------------------   
    
    (2) Adversarial Optimization
    -----------------------------------------------------------------------------------------
    | Pointwise | IRGAN_Point                                                               |
    -----------------------------------------------------------------------------------------
    | Pairwise  | IRGAN_Pair                                                                |
    -----------------------------------------------------------------------------------------
    | Listwise  | IRGAN_List                                                                |
    -----------------------------------------------------------------------------------------
    
    (3) Tree-based Model (provided by LightGBM)
    -----------------------------------------------------------------------------------------
    | LightGBMLambdaMART                                                                    |
    -----------------------------------------------------------------------------------------
    

    >>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_LTR | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_LTR | Istella_S | Istella | Istella_X                                         |
    -----------------------------------------------------------------------------------------

    """

    parser = argparse.ArgumentParser('Run pt_ranking.')
    parser.add_argument('-cuda', type=int, help='specify the gpu id if needed, such as 0 or 1.', default=None)

    ''' path of json files specifying the evaluation details '''
    parser.add_argument('-dir_json', type=str, help='the path of json files specifying the evaluation details.')

    ''' learning rate for pretraining'''
    parser.add_argument('-pretrain_lr', type=float, help='the learning rate for pretraining.')

    ''' path of json files specifying the evaluation details '''
    parser.add_argument('-finetune_lr', type=float, help='the learning rate for finetuning.')

    ''' the output_dir '''
    parser.add_argument('-trial_num', type=int, help='the trial number.')

    ''' the augmentation '''
    parser.add_argument('-aug_type', type=str, help='the type of augmentation.')

    ''' the augmentation percentage '''
    parser.add_argument('-aug_percent', type=float, help='the percentage of augmentation.')

    ''' simsiam dim '''
    parser.add_argument('-dim', type=int, help='the dimension of pretrainer.')

    ''' layers '''
    parser.add_argument('-layers', type=int, help='the number of layers.')

    ''' pretrainer '''
    parser.add_argument('-pretrainer', type=str, help='the type of pretrainer.')

    ''' temp '''
    parser.add_argument('-temperature', type=float, help='temperature of the softmax loss (if applicable).')

    ''' mix '''
    parser.add_argument('-mix', type=float, help='Mix between instance and set level simclr')

    ''' shrink '''
    parser.add_argument('-shrink', type=float, help='How much to shrink the train set')

    ''' blend '''
    parser.add_argument('-blend', type=float, help='Mixing between instance and qg level')

    ''' scale '''
    parser.add_argument('-scale', type=float, help='Scale of gaussian')

    ''' gumbel '''
    parser.add_argument('-gumbel', type=float, help='Temperature scaling of gumbel noise')

    ''' num negatives '''
    parser.add_argument('-num_negatives', type=int, help='Number of negatives per qg')
    
    ''' freeze '''
    parser.add_argument('-freeze', type=int, help='Whether to freeze prev layers')

    ''' number of probing layers '''
    parser.add_argument('-probe_layers', type=int, help='Number of probing layers')

    ''' finetune_only '''
    parser.add_argument('-finetune_only', type=int, help='no pretrain')

    ''' finetune_trial '''
    parser.add_argument('-finetune_trials', type=float, help='finetune_trials')

    argobj = parser.parse_args()
    
    if argobj.pretrainer == 'LightGBMLambdaMART':
        evaluator = TreeLTREvaluator()
        evaluator.run(model_id=argobj.pretrainer, dir_json=argobj.dir_json, config_with_json=True, argobj=argobj)

    else:
        if argobj.freeze:
            print('Finetune FROZEN')
        evaluator = LTREvaluator(cuda=argobj.cuda)
        
        file_pretrain = argobj.pretrainer
        if argobj.pretrainer == 'SubTab' or argobj.pretrainer == 'VIME':
            file_pretrain = 'RankNeg'

        # setup_seed(0)
        if argobj.aug_type != 'none' and not argobj.finetune_only:
            print('Starting pretraining!', sys.stderr)
            argobj.is_pretraining = True
            evaluator.run(model_id=file_pretrain, dir_json=os.path.join(argobj.dir_json, '{0}/'.format(argobj.pretrainer)), config_with_json=True, argobj=argobj)

        print('Starting finetuning!', sys.stderr)
        argobj.is_pretraining = False
        if argobj.aug_type == 'none':
            evaluator.run(model_id="LambdaRank", dir_json=os.path.join(argobj.dir_json, 'lambdarank/'), config_with_json=True, argobj=argobj)
        elif argobj.pretrainer == 'SubTab':
            evaluator.run(model_id="SubTabTune", dir_json=os.path.join(argobj.dir_json, 'lambdaranktune/'), config_with_json=True, argobj=argobj)
        else:
            evaluator.run(model_id="LambdaRankTune", dir_json=os.path.join(argobj.dir_json, 'lambdaranktune/'), config_with_json=True, argobj=argobj)

