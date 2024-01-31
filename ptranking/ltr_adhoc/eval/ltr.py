#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Description
A general framework for evaluating traditional learning-to-rank methods.
"""

import os
import sys
import datetime
import numpy as np
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

from ptranking.base.ranker import LTRFRAME_TYPE
from ptranking.metric.metric_utils import metric_results_to_string
from ptranking.data.data_utils import SPLIT_TYPE, LABEL_TYPE, LETORSampler
from ptranking.data.MSLR_dataset_filters import mslr_filters
from ptranking.data.yahoo1_dataset_filters import set1_filters
from ptranking.data.set2_dataset_filters import set2_filters
from ptranking.data.istella_filters import istella_filters
from ptranking.data.data_utils import LTRDataset, YAHOO_LTR, ISTELLA_LTR, MSLETOR_SEMI, MSLETOR_LIST
from ptranking.ltr_adhoc.eval.parameter import ModelParameter, DataSetting, EvalSetting, ScoringFunctionParameter, ValidationTape, CVTape, SummaryTape, OptLossTape

from ptranking.ltr_adhoc.pointwise.rank_mse import RankMSE
from ptranking.ltr_adhoc.pairwise.ranknet import RankNet, RankNetParameter
from ptranking.ltr_adhoc.listwise.lambdarank import LambdaRank, LambdaRankParameter
from ptranking.ltr_adhoc.listwise.listnet import ListNet
from ptranking.ltr_adhoc.listwise.listmle import ListMLE
from ptranking.ltr_adhoc.listwise.rank_cosine import RankCosine
from ptranking.ltr_adhoc.listwise.approxNDCG import ApproxNDCG, ApproxNDCGParameter
from ptranking.ltr_adhoc.listwise.wassrank.wassRank import WassRank, WassRankParameter
from ptranking.ltr_adhoc.listwise.st_listnet import STListNet, STListNetParameter
from ptranking.ltr_adhoc.listwise.lambdaloss import LambdaLoss, LambdaLossParameter
from ptranking.ltr_adhoc.pretrain.simsiam import SimSiam, SimSiamParameter
from ptranking.ltr_adhoc.pretrain.simrank import SimRank, SimRankParameter
from ptranking.ltr_adhoc.pretrain.simclr import SimCLR, SimCLRParameter
from ptranking.ltr_adhoc.pretrain.simsiam_rank import SimSiamRank, SimSiamRankParameter
from ptranking.ltr_adhoc.listwise.lambdaranktune import LambdaRankTune, LambdaRankTuneParameter
from ptranking.ltr_adhoc.listwise.subtab_tune import SubTabTune, SubTabTuneParameter
from ptranking.ltr_adhoc.pretrain.rankneg import RankNeg, RankNegParameter

LTR_ADHOC_MODEL = [
    'RankMSE', 'RankNet', 'RankCosine', 'ListNet', 'ListMLE', 'STListNet',
    'ApproxNDCG', 'WassRank', 'LambdaRank', 'SoftRank', 'LambdaLoss',
    'TwinRank', 'SimSiam', 'LambdaRankTune', 'SimRank', 'SimCLR', 'SimSiamRank', 'RankNeg', 'SubTabTune'
]


class LTREvaluator():
    """
    The class for evaluating different ltr_adhoc methods.
    """

    def __init__(self, frame_id=LTRFRAME_TYPE.Adhoc, cuda=None):
        self.frame_id = frame_id

        if cuda is None:
            self.gpu, self.device = False, 'cpu'
        else:
            self.gpu, self.device = True, 'cuda:' + str(cuda)
            torch.cuda.set_device(cuda)

    def display_information(self, data_dict, model_para_dict, reproduce=False):
        """
        Display some information.
        :param data_dict:
        :param model_para_dict:
        :return:
        """
        if self.gpu:
            print('-- GPU({}) is launched --'.format(self.device))
        else:
            print('Only CPU is used.')

        if reproduce:
            print(' '.join([
                '\nReproducing results for {} on {} >>>'.format(
                    model_para_dict['model_id'], data_dict['data_id'])
            ]))
        else:
            print(' '.join([
                '\nStart {} on {} >>>'.format(model_para_dict['model_id'],
                                              data_dict['data_id'])
            ]))

    def check_consistency(self, data_dict, eval_dict, sf_para_dict):
        """
        Check whether the settings are reasonable in the context of adhoc learning-to-rank
        """
        ''' Part-1: data loading '''

        if data_dict['data_id'] == 'Istella':
            assert eval_dict[
                'do_validation'] is not True  # since there is no validation data

        if data_dict['data_id'] in MSLETOR_SEMI:
            assert data_dict[
                'train_presort'] is not True  # due to the non-labeled documents
            if data_dict[
                    'binary_rele']:  # for unsupervised dataset, it is required for binarization due to '-1' labels
                assert data_dict['unknown_as_zero']
        else:
            assert data_dict[
                'unknown_as_zero'] is not True  # since there is no non-labeled documents

        if data_dict[
                'data_id'] in MSLETOR_LIST:  # for which the standard ltr_adhoc of each query is unique
            assert 1 == data_dict['train_batch_size']

        if data_dict['scale_data']:
            scaler_level = data_dict[
                'scaler_level'] if 'scaler_level' in data_dict else None
            assert not scaler_level == 'DATASET'  # not supported setting

        assert data_dict[
            'validation_presort']  # Rule of thumb setting for adhoc learning-to-rank
        assert data_dict[
            'test_presort']  # Rule of thumb setting for adhoc learning-to-rank
        ''' Part-2: evaluation setting '''

        if eval_dict[
                'mask_label']:  # True is aimed to use supervised data to mimic semi-supervised data by masking
            assert not data_dict['data_id'] in MSLETOR_SEMI

    def determine_files(self, data_dict, fold_k=None):
        """
        Determine the file path correspondingly.
        :param data_dict:
        :param fold_k:
        :return:
        """
        if data_dict['data_id'] in YAHOO_LTR:
            file_train, file_vali, file_test = os.path.join(data_dict['dir_data'], data_dict['data_id'].lower() + '.train.txt'),\
                                               os.path.join(data_dict['dir_data'], data_dict['data_id'].lower() + '.valid.txt'),\
                                               os.path.join(data_dict['dir_data'], data_dict['data_id'].lower() + '.test.txt')

        elif data_dict['data_id'] in ISTELLA_LTR:
            if data_dict['data_id'] == 'Istella_X' or data_dict[
                    'data_id'] == 'Istella_S':
                file_train, file_vali, file_test = data_dict[
                    'dir_data'] + 'train.txt', data_dict[
                        'dir_data'] + 'vali.txt', data_dict[
                            'dir_data'] + 'test.txt'
            else:
                file_vali = None
                file_train, file_test = data_dict[
                    'dir_data'] + 'train.txt', data_dict[
                        'dir_data'] + 'test.txt'
        else:
            print('Fold-', fold_k)
            fold_k_dir = data_dict['dir_data'] + 'Fold' + str(fold_k) + '/'
            file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'

        return file_train, file_vali, file_test

    def load_data(self,
                  eval_dict,
                  data_dict,
                  fold_k,
                  shrink):
        """
        Load the dataset correspondingly.
        :param eval_dict:
        :param data_dict:
        :param fold_k:
        :param model_para_dict:
        :return:
        """
        file_train, file_vali, file_test = self.determine_files(data_dict,
                                                                fold_k=fold_k)

        input_eval_dict = eval_dict if eval_dict[
            'mask_label'] else None  # required when enabling masking data

        _train_data = LTRDataset(file=file_train,
                                 split_type=SPLIT_TYPE.Train,
                                 presort=data_dict['train_presort'],
                                 data_dict=data_dict,
                                 eval_dict=input_eval_dict)
        print('Previous train size', len(_train_data.list_torch_Qs))
        _train_data.list_torch_Qs = _train_data.list_torch_Qs[:int(len(_train_data.list_torch_Qs) * shrink)]
        print('New train size', len(_train_data.list_torch_Qs))
        train_letor_sampler = LETORSampler(
            data_source=_train_data,
            rough_batch_size=data_dict['train_rough_batch_size'])
        train_loader = torch.utils.data.DataLoader(
            _train_data, batch_sampler=train_letor_sampler, num_workers=0)

        _test_data = LTRDataset(file=file_test,
                                split_type=SPLIT_TYPE.Test,
                                data_dict=data_dict,
                                presort=data_dict['test_presort'])
        test_letor_sampler = LETORSampler(
            data_source=_test_data,
            rough_batch_size=data_dict['test_rough_batch_size'])
        test_loader = torch.utils.data.DataLoader(
            _test_data, batch_sampler=test_letor_sampler, num_workers=0)

        if eval_dict['do_validation'] or eval_dict[
                'do_summary']:  # vali_data is required
            _vali_data = LTRDataset(file=file_vali,
                                    split_type=SPLIT_TYPE.Validation,
                                    data_dict=data_dict,
                                    presort=data_dict['validation_presort'])
            vali_letor_sampler = LETORSampler(
                data_source=_vali_data,
                rough_batch_size=data_dict['validation_rough_batch_size'])
            vali_loader = torch.utils.data.DataLoader(
                _vali_data, batch_sampler=vali_letor_sampler, num_workers=0)
        else:
            vali_loader = None

        return train_loader, test_loader, vali_loader

    def load_ranker(self, sf_para_dict, model_para_dict):
        """
        Load a ranker correspondingly
        :param sf_para_dict:
        :param model_para_dict:
        :param kwargs:
        :return:
        """
        model_id = model_para_dict['model_id']

        if model_id in [
                'RankMSE', 'ListMLE', 'ListNet', 'RankCosine', 'DASALC',
                'HistogramAP'
        ]:
            ranker = globals()[model_id](sf_para_dict=sf_para_dict,
                                         gpu=self.gpu,
                                         device=self.device)

        elif model_id in [
                'RankNet', 'LambdaRank', 'STListNet', 'ApproxNDCG',
                'DirectOpt', 'LambdaLoss', 'MarginLambdaLoss', 'MDPRank',
                'ExpectedUtility', 'MDNExpectedUtility', 'RankingMDN',
                'SoftRank', 'TwinRank', 'SimSiam', 'LambdaRankTune', 'SimRank', 'SimCLR', 'SimSiamRank', 'RankNeg', 'SubTabTune'
        ]:
            ranker = globals()[model_id](sf_para_dict=sf_para_dict,
                                         model_para_dict=model_para_dict,
                                         gpu=self.gpu,
                                         device=self.device)

        elif model_id == 'WassRank':
            ranker = WassRank(sf_para_dict=sf_para_dict,
                              wass_para_dict=model_para_dict,
                              dict_cost_mats=self.dict_cost_mats,
                              dict_std_dists=self.dict_std_dists,
                              gpu=self.gpu,
                              device=self.device)
        else:
            raise NotImplementedError

        return ranker

    def setup_output(self, data_dict=None, eval_dict=None):
        """
        Update output directory
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        model_id = self.model_parameter.model_id
        grid_search, do_vali, dir_output = eval_dict['grid_search'], eval_dict[
            'do_validation'], eval_dict['dir_output']
        mask_label = eval_dict['mask_label']

        if grid_search:
            dir_root = dir_output + '_'.join([
                'gpu', 'grid', model_id
            ]) + '/' if self.gpu else dir_output + '_'.join(['grid', model_id
                                                             ]) + '/'
        else:
            dir_root = dir_output

        eval_dict['dir_root'] = dir_root
        if not os.path.exists(dir_root): os.makedirs(dir_root)

        sf_str = self.sf_parameter.to_para_string()
        data_eval_str = '_'.join([
            self.data_setting.to_data_setting_string(),
            self.eval_setting.to_eval_setting_string()
        ])
        if mask_label:
            data_eval_str = '_'.join([
                data_eval_str, 'MaskLabel', 'Ratio',
                '{:,g}'.format(eval_dict['mask_ratio'])
            ])

        file_prefix = '_'.join([model_id, 'SF', sf_str, data_eval_str])

        if data_dict['scale_data']:
            if data_dict['scaler_level'] == 'QUERY':
                file_prefix = '_'.join(
                    [file_prefix, 'QS', data_dict['scaler_id']])
            else:
                file_prefix = '_'.join(
                    [file_prefix, 'DS', data_dict['scaler_id']])

        dir_run = dir_root + file_prefix + '/'  # run-specific outputs

        model_para_string = self.model_parameter.to_para_string()
        if len(model_para_string) > 0:
            dir_run = dir_run + model_para_string + '/'

        eval_dict['dir_run'] = dir_run
        if not os.path.exists(dir_run):
            os.makedirs(dir_run)

        return dir_run

    def setup_eval(self, data_dict, eval_dict, sf_para_dict, model_para_dict):
        """
        Finalize the evaluation setting correspondingly
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        sf_para_dict[sf_para_dict['sf_id']].update(
            dict(num_features=data_dict['num_features']))

        #self.dir_run  = self.setup_output(data_dict, eval_dict)
        eval_dict['dir_root'] = eval_dict['dir_output']
        self.dir_run = eval_dict['dir_output']

        if eval_dict['do_log'] and not self.eval_setting.debug:
            time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
            sys.stdout = open(
                self.dir_run + '_'.join(['log', time_str]) + '.txt', "w")

        #if self.do_summary: self.summary_writer = SummaryWriter(self.dir_run + 'summary')
        if not model_para_dict['model_id'] in [
                'MDPRank', 'ExpectedUtility', 'WassRank'
        ]:
            """
            Aiming for efficient batch processing, please use a large batch_size, e.g., {train_rough_batch_size, validation_rough_batch_size, test_rough_batch_size = 300, 300, 300}
            """
            #assert data_dict['train_rough_batch_size'] > 1

    def log_max(self,
                data_dict=None,
                max_cv_avg_scores=None,
                sf_para_dict=None,
                eval_dict=None,
                log_para_str=None):
        ''' Log the best performance across grid search and the corresponding setting '''
        dir_root, cutoffs = eval_dict['dir_root'], eval_dict['cutoffs']
        data_id = data_dict['data_id']

        sf_str = self.sf_parameter.to_para_string(log=True)

        data_eval_str = self.data_setting.to_data_setting_string(
            log=True) + '\n' + self.eval_setting.to_eval_setting_string(
                log=True)

        with open(file=dir_root + '/' +
                  '_'.join([data_id, sf_para_dict['sf_id'], 'max.txt']),
                  mode='w') as max_writer:
            max_writer.write('\n\n'.join([
                data_eval_str, sf_str, log_para_str,
                metric_results_to_string(max_cv_avg_scores,
                                         cutoffs,
                                         metric='nDCG')
            ]))

    def kfold_cv_reproduce(self,
                           data_dict=None,
                           eval_dict=None,
                           sf_para_dict=None,
                           model_para_dict=None):
        self.display_information(data_dict, model_para_dict)
        self.check_consistency(data_dict, eval_dict, sf_para_dict)

        model_id = model_para_dict['model_id']
        fold_num, max_label = data_dict['fold_num'], data_dict[
            'max_rele_level']
        cutoffs, do_vali = eval_dict['cutoffs'], eval_dict['do_validation']

        cv_tape = CVTape(model_id=model_id,
                         fold_num=fold_num,
                         cutoffs=cutoffs,
                         do_validation=do_vali,
                         reproduce=True)

        sf_para_dict[sf_para_dict['sf_id']].update(
            dict(num_features=data_dict['num_features']))
        ranker = self.load_ranker(model_para_dict=model_para_dict,
                                  sf_para_dict=sf_para_dict)

        model_exp_dir = self.setup_output(data_dict, eval_dict)
        for fold_k in range(1, fold_num + 1):  # evaluation over k-fold data
            ranker.init(
            )  # initialize or reset with the same random initialization

            _, test_data, _ = self.load_data(eval_dict, data_dict, fold_k)

            cv_tape.fold_evaluation_reproduce(ranker=ranker,
                                              test_data=test_data,
                                              dir_run=model_exp_dir,
                                              max_label=max_label,
                                              fold_k=fold_k,
                                              model_id=model_id,
                                              device=self.device)

        ndcg_cv_avg_scores = cv_tape.get_cv_performance()
        return ndcg_cv_avg_scores

    def update_dicts(self, data_dict, eval_dict, sf_para_dict, model_para_dict,
                     argobj):
        if argobj.aug_type == 'none':
            eval_dict['dir_output'] = os.path.join(
            eval_dict['dir_output'],
            'Scratch_{0}_layers{2}_trial{1}_shrink{3}/'.format(
                argobj.finetune_lr, argobj.trial_num, argobj.layers, argobj.shrink))
        else:
            if argobj.pretrainer == 'SimSiam':
                eval_dict['dir_output'] = os.path.join(
                    eval_dict['dir_output'],
                    'SimSiam_{0}{1}_{2}_dim_{5}_layers_{6}_to_finetune_{3}_trial{4}_shrink0.01/'.format(
                        argobj.aug_type, argobj.aug_percent, argobj.pretrain_lr,
                        argobj.finetune_lr, argobj.trial_num, argobj.dim, argobj.layers, argobj.shrink))
            elif argobj.pretrainer == 'SimRank' or argobj.pretrainer == 'SimCLR' or argobj.pretrainer == 'SimSiamRank' or argobj.pretrainer == 'RankNeg' or argobj.pretrainer == 'SubTab' or argobj.pretrainer == 'VIME':
                eval_dict['dir_output'] = os.path.join(
                    eval_dict['dir_output'],
                    '{9}_{0}{1}_{2}_dim_{5}_layers_{6}_to_finetune_{3}_temp{7}_mix0.0_trial{4}_shrink0.01_blend{11}_scale{12}_gumbel{13}_numnegatives{14}/'.format(
                        argobj.aug_type, argobj.aug_percent, argobj.pretrain_lr,
                        argobj.finetune_lr, argobj.trial_num, argobj.dim, argobj.layers, argobj.temperature, argobj.mix, argobj.pretrainer, argobj.shrink, argobj.blend, argobj.scale, argobj.gumbel, argobj.num_negatives))
            else:
                raise ValueError('Should be one of SimSiam or SimRank or SimCLR or SimSiamRank')
        if not os.path.exists(eval_dict['dir_output']):
            os.makedirs(eval_dict['dir_output'])
        
        if argobj.is_pretraining:
            if argobj.pretrainer == 'SimSiam':
                sf_para_dict['lr'] = argobj.pretrain_lr
                sf_para_dict['layers'] = argobj.layers
                model_para_dict['aug_type'] = argobj.aug_type
                model_para_dict['aug_percent'] = argobj.aug_percent
                model_para_dict['dim'] = argobj.dim 
            elif argobj.pretrainer == 'SimRank' or argobj.pretrainer == 'SimCLR' or argobj.pretrainer == 'SimSiamRank' or argobj.pretrainer == 'RankNeg' or argobj.pretrainer == 'VIME' or argobj.pretrainer == 'SubTab':
                sf_para_dict['lr'] = argobj.pretrain_lr
                sf_para_dict['layers'] = argobj.layers
                model_para_dict['aug_type'] = argobj.aug_type
                model_para_dict['aug_percent'] = argobj.aug_percent
                model_para_dict['dim'] = argobj.dim 
                model_para_dict['temp'] = argobj.temperature
                model_para_dict['mix'] = argobj.mix
                model_para_dict['blend'] = argobj.blend
                model_para_dict['scale'] = argobj.scale
                model_para_dict['gumbel'] = argobj.gumbel
                model_para_dict['num_negatives'] = argobj.num_negatives
            else:
                raise ValueError('Should be one of SimSiam or SimRank')
        else:
            sf_para_dict['lr'] = argobj.finetune_lr
            model_para_dict['freeze'] = argobj.freeze
            model_para_dict['probe_layers'] = argobj.probe_layers
            model_para_dict['gumbel'] = argobj.gumbel
            if argobj.aug_type != 'none' or argobj.freeze:
                model_para_dict['model_path'] = eval_dict['dir_output']
        print(eval_dict)
        
    def kfold_cv_eval(self,
                      data_dict=None,
                      eval_dict=None,
                      sf_para_dict=None,
                      model_para_dict=None,
                      argobj=None):
        """
        Evaluation learning-to-rank methods via k-fold cross validation if there are k folds, otherwise one fold.
        :param data_dict:       settings w.r.t. data
        :param eval_dict:       settings w.r.t. evaluation
        :param sf_para_dict:    settings w.r.t. scoring function
        :param model_para_dict: settings w.r.t. the ltr_adhoc model
        :return:
        """
        self.update_dicts(data_dict, eval_dict, sf_para_dict, model_para_dict,
                          argobj)
        self.display_information(data_dict, model_para_dict)
        self.check_consistency(data_dict, eval_dict, sf_para_dict)

        ranker = self.load_ranker(model_para_dict=model_para_dict,
                                  sf_para_dict=sf_para_dict)
        print('Loading ranker', file=sys.stderr)
        ranker.uniform_eval_setting(eval_dict=eval_dict)

        self.setup_eval(data_dict, eval_dict, sf_para_dict, model_para_dict)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.dir_run + current_time
        summary_writer = SummaryWriter(log_dir)
        print('Tensorboard dir: ' + log_dir, file=sys.stderr)

        model_id = model_para_dict['model_id']
        fold_num, label_type, max_label = data_dict['fold_num'], data_dict[
            'label_type'], data_dict['max_rele_level']
        train_presort, validation_presort, test_presort = \
            data_dict['train_presort'], data_dict['validation_presort'], data_dict['test_presort']
        # for quick access of common evaluation settings
        epochs, loss_guided = eval_dict['epochs'], eval_dict['loss_guided']
        vali_k, log_step, cutoffs = eval_dict['vali_k'], eval_dict[
            'log_step'], eval_dict['cutoffs']
        do_vali, vali_metric, do_summary = eval_dict[
            'do_validation'], eval_dict['vali_metric'], eval_dict['do_summary']

        cv_tape = CVTape(model_id=model_id,
                         fold_num=fold_num,
                         cutoffs=cutoffs,
                         do_validation=do_vali)
        traintape = CVTape(model_id=model_id,
                           fold_num=fold_num,
                           cutoffs=cutoffs,
                           do_validation=do_vali)
        for fold_k in range(1, 2):  # evaluation over k-fold data
            ranker.init(
            )  # initialize or reset with the same random initialization
            print('Initialized the model, loading data', file=sys.stderr)
            if model_id not in ['SimRank', 'SimSiam', 'SimCLR', 'SimSiamRank', 'RankNeg']:
                train_data, test_data, vali_data = self.load_data(
                    eval_dict, data_dict, fold_k, argobj.shrink)
            else:
                train_data, test_data, vali_data = self.load_data(
                    eval_dict, data_dict, fold_k, 1.0)
            if do_vali:
                vali_tape = ValidationTape(fold_k=fold_k,
                                           num_epochs=epochs,
                                           validation_metric=vali_metric,
                                           validation_at_k=vali_k,
                                           dir_run=self.dir_run)
            if do_summary:
                summary_tape = SummaryTape(do_validation=do_vali,
                                           cutoffs=cutoffs,
                                           label_type=label_type,
                                           train_presort=train_presort,
                                           test_presort=test_presort,
                                           gpu=self.gpu)
            if not do_vali and loss_guided:
                opt_loss_tape = OptLossTape(gpu=self.gpu)

            best_metric_val = None
            for epoch_k in range(1, epochs + 1):
                torch_fold_k_epoch_k_loss, stop_training = ranker.train(
                        train_data=train_data,
                        epoch_k=epoch_k,
                        presort=train_presort,
                        label_type=label_type)
                print('epoch', epoch_k, 'train loss', torch_fold_k_epoch_k_loss, file=sys.stderr)
                train_loss_metric_val = torch_fold_k_epoch_k_loss.squeeze(
                        -1).data.cpu().numpy()
                summary_writer.add_scalar('train loss', train_loss_metric_val, global_step=epoch_k)
                if model_id not in ['SimSiam', 'SimRank', 'SimCLR', 'SimSiamRank', 'RankNeg']:
                    train_loss_metric_val = torch_fold_k_epoch_k_loss.squeeze(
                        -1).data.cpu().numpy()
                    torch_train_metric_value = ranker.validation(
                        vali_data=train_data,
                        k=vali_k,
                        device=ranker.device,
                        vali_metric=vali_metric,
                        label_type=label_type,
                        max_label=max_label,
                        presort=train_presort)
                    train_ndcg_print = torch_train_metric_value.squeeze().data.cpu(
                    ).numpy()
                    val_train_metric_value = ranker.validation(
                        vali_data=vali_data,
                        k=vali_k,
                        device=ranker.device,
                        vali_metric=vali_metric,
                        label_type=label_type,
                        max_label=max_label,
                        presort=validation_presort)
                    val_ndcg_print = val_train_metric_value.squeeze().data.cpu(
                    ).numpy()
                    summary_writer.add_scalar('train_ndcg',
                                        train_ndcg_print,
                                        global_step=epoch_k)
                    summary_writer.add_scalar('val_ndcg', val_ndcg_print, global_step=epoch_k)
                    print(
                        'Fold {0}   Epoch {1}   Loss {2}  Train {3} Val {4}'
                        .format(fold_k, epoch_k, torch_fold_k_epoch_k_loss,
                                train_ndcg_print, val_ndcg_print),
                        file=sys.stderr)

                    if best_metric_val is None or val_ndcg_print > best_metric_val:
                        best_metric_val = val_ndcg_print
                        ranker.save(dir=self.dir_run + '/',
                                    name='_'.join(['net_params_best', str(argobj.freeze), str(argobj.probe_layers), str(argobj.finetune_only), str(argobj.finetune_trials), str(argobj.shrink)]))
                    # with summary_writer.as_default():
                    #     tf.summary.scalar('train_loss',
                    #                     train_loss_metric_val,
                    #                     step=epoch_k)
                    summary_writer.add_scalar('train_loss',
                                        train_loss_metric_val,
                                        global_step=epoch_k)
                ranker.scheduler.step(
                )  # adaptive learning rate with step_size=40, gamma=0.5
                

                if stop_training:
                    print('training is failed !')
                    break
            print("Saving in..", self.dir_run, file = sys.stderr)
            if model_id in ['SimSiam', 'SimRank', 'SimCLR', 'SimSiamRank', 'RankNeg']:
                ranker.save(dir=self.dir_run + '/',
                        name='_'.join(['net_params_pretrain']))
            ranker.save(dir=self.dir_run + '/',
                        name='_'.join(['net_params']) + '.pkl')

            if do_summary:  # track
                summary_tape.fold_summary(
                    fold_k=fold_k,
                    dir_run=self.dir_run,
                    train_data_length=train_data.__len__())
            if model_id not in ['SimSiam', 'SimRank', 'SimCLR', 'SimSiamRank', 'RankNeg']:
                ranker.load(self.dir_run + '/' + '_'.join(['net_params_best', str(argobj.freeze), str(argobj.probe_layers), str(argobj.finetune_only), str(argobj.finetune_trials), str(argobj.shrink)]),
                            device=self.device)
            else:
                ranker.load(self.dir_run + '/' + '_'.join(['net_params']) +
                        '.pkl',
                        device=self.device)
            
            print('Finetuned model saved in', '_'.join(['net_params_best', str(argobj.freeze), str(argobj.probe_layers), str(argobj.finetune_only), str(argobj.finetune_trials), str(argobj.shrink)]), file=sys.stderr)
            
            # if do_vali: # using the fold-wise optimal model for later testing based on validation data
            #     ranker.load(modeldir, device=self.device)
            #vali_tape.clear_fold_buffer(fold_k=fold_k)
            # else:            # buffer the model after a fixed number of training-epoches if no validation is deployed
            #     fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
            #     ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')

            hparam_dict = {                           
                            'pretrainer': argobj.pretrainer,    
                            'aug_percent': argobj.aug_percent,
                           'blend': argobj.blend,
                           'gumbel': argobj.gumbel,
                           'num_negatives': argobj.num_negatives,
                            'pretrain_lr': argobj.pretrain_lr,
                           'finetune_lr': argobj.finetune_lr,
                           'trial': argobj.trial_num,
                           'scale': argobj.scale
            }
            metrics_dict = {}
            if data_dict['data_id'] == 'MSLRWEB30K':
                curr_filters = mslr_filters
            elif data_dict['data_id'] == 'Set1':
                curr_filters = set1_filters
            elif data_dict['data_id'] == 'Set2':
                curr_filters = set2_filters
            elif data_dict['data_id'] == 'Istella_S':
                curr_filters = istella_filters

            if model_id not in ['SimRank', 'SimSiam', 'SimCLR', 'SimSiamRank', 'RankNeg']:
                # print('Train results')
                # ndcgs, ks = traintape.fold_evaluation(model_id=model_id,
                #                         ranker=ranker,
                #                         test_data=small_train_set,
                #                         max_label=max_label,
                #                         fold_k=fold_k)
                # metrics_dict['train/ndcg@3'] = ndcgs[1]
                # metrics_dict['train/ndcg@5'] = ndcgs[2]
                # metrics_dict['train/ndcg@10'] = ndcgs[3]
                # metrics_dict['train/ndcg@20'] = ndcgs[4]
                # if do_vali:
                    # print('Val results')
                    # ndcgs, ks = cv_tape.fold_evaluation(model_id=model_id,
                    #                         ranker=ranker,
                    #                         test_data=vali_data,
                    #                         max_label=max_label,
                    #                         fold_k=fold_k)
                    
                    # summary_writer.add_hparams(hparam_dict, {'hparam/val/ndcg@3': ndcgs[1], 'hparam/val/ndcg@5': ndcgs[2], 'hparam/val/ndcg@10': ndcgs[3],'hparam/val/ndcg@20': ndcgs[4]})
                    # metrics_dict['val/ndcg@3'] = ndcgs[1]
                    # metrics_dict['val/ndcg@5'] = ndcgs[2]
                    # metrics_dict['val/ndcg@10'] = ndcgs[3]
                    # metrics_dict['val/ndcg@20'] = ndcgs[4]

                    # print('Overall robust val results')
                    # ndcgs, ks = cv_tape.fold_evaluation(model_id=model_id,
                    #                     ranker=ranker,
                    #                     test_data=vali_data,
                    #                     max_label=max_label,
                    #                     fold_k=fold_k,
                    #                     filters=curr_filters)

                    # summary_writer.add_hparams(hparam_dict, {'hparam/val/robust-ndcg@3': ndcgs[1], 'hparam/val/robust-ndcg@5': ndcgs[2], 'hparam/val/robust-ndcg@10': ndcgs[3],'hparam/val/robust-ndcg@20': ndcgs[4]})
                    # metrics_dict['val/robust-ndcg@3'] = ndcgs[1]
                    # metrics_dict['val/robust-ndcg@5'] = ndcgs[2]
                    # metrics_dict['val/robust-ndcg@10'] = ndcgs[3]
                    # metrics_dict['val/robust-ndcg@20'] = ndcgs[4]
                # print('Individual robust val results')
                # for curr_filter in filters:
                #     print('Robust dim {0}'.format(curr_filter[0]))
                #     cv_tape.fold_evaluation(model_id=model_id,
                #                         ranker=ranker,
                #                         test_data=vali_data,
                #                         max_label=max_label,
                #                         fold_k=fold_k,
                #                         filters=[curr_filter])

                print('Test results')
                ndcgs, ks = cv_tape.fold_evaluation(model_id=model_id,
                                        ranker=ranker,
                                        test_data=test_data,
                                        max_label=max_label,
                                        fold_k=fold_k)

                summary_writer.add_hparams(hparam_dict, {'hparam/test/ndcg@3': ndcgs[1], 'hparam/test/ndcg@5': ndcgs[2], 'hparam/test/ndcg@10': ndcgs[3],'hparam/test/ndcg@20': ndcgs[4]})
                metrics_dict['test/ndcg@3'] = ndcgs[1]
                metrics_dict['test/ndcg@5'] = ndcgs[2]
                metrics_dict['test/ndcg@10'] = ndcgs[3]
                metrics_dict['test/ndcg@20'] = ndcgs[4]

                print('Overall robust test results')
                ndcgs, ks =  cv_tape.fold_evaluation(model_id=model_id,
                                    ranker=ranker,
                                    test_data=test_data,
                                    max_label=max_label,
                                    fold_k=fold_k,
                                    filters=curr_filters)
                metrics_dict['test/robust-ndcg@3'] = ndcgs[1]
                metrics_dict['test/robust-ndcg@5'] = ndcgs[2]
                metrics_dict['test/robust-ndcg@10'] = ndcgs[3]
                metrics_dict['test/robust-ndcg@20'] = ndcgs[4]

                # summary_writer.add_hparams(hparam_dict, {'hparam/test/robust-ndcg@3': ndcgs[1], 'hparam/test/robust-ndcg@5': ndcgs[2], 'hparam/test/robust-ndcg@10': ndcgs[3],'hparam/test/robust-ndcg@20': ndcgs[4]})

                # print('Individual robust val results')
                # for curr_filter in filters:
                #     print('Robust dim {0}'.format(curr_filter[0]))
                #     cv_tape.fold_evaluation(model_id=model_id,
                #                         ranker=ranker,
                #                         test_data=test_data,
                #                         max_label=max_label,
                #                         fold_k=fold_k,
                #                         filters=[curr_filter])
                
                with open(self.dir_run + 'hparam.pickle', 'wb') as handle:
                    pickle.dump(hparam_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                with open(self.dir_run + '_'.join(['metrics', str(argobj.freeze), str(argobj.probe_layers), str(argobj.finetune_only), str(argobj.finetune_trials), str(argobj.shrink)]) + '.pickle', 'wb') as handle:
                    pickle.dump(metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Tensorboard dir: ' + log_dir, file=sys.stderr)

    def naive_train(self, ranker, eval_dict, train_data=None, test_data=None):
        """
        A simple train and test, namely train based on training data & test based on testing data
        :param ranker:
        :param eval_dict:
        :param train_data:
        :param test_data:
        :param vali_data:
        :return:
        """
        ranker.reset_parameters()  # reset with the same random initialization

        assert train_data is not None
        assert test_data is not None

        list_losses = []
        list_train_ndcgs = []
        list_test_ndcgs = []

        epochs, cutoffs = eval_dict['epochs'], eval_dict['cutoffs']

        for i in range(epochs):
            epoch_loss = torch.zeros(1).to(
                self.device) if self.gpu else torch.zeros(1)
            for qid, batch_rankings, batch_stds in train_data:
                if self.gpu:
                    batch_rankings, batch_stds = batch_rankings.to(
                        self.device), batch_stds.to(self.device)
                batch_loss, stop_training = ranker.train(batch_rankings,
                                                         batch_stds,
                                                         qid=qid)
                epoch_loss += batch_loss.item()

            np_epoch_loss = epoch_loss.cpu().numpy(
            ) if self.gpu else epoch_loss.data.numpy()
            list_losses.append(np_epoch_loss)

            test_ndcg_ks = ranker.ndcg_at_ks(test_data=test_data,
                                             ks=cutoffs,
                                             label_type=LABEL_TYPE.MultiLabel,
                                             device='cpu')
            np_test_ndcg_ks = test_ndcg_ks.data.numpy()
            list_test_ndcgs.append(np_test_ndcg_ks)

            train_ndcg_ks = ranker.ndcg_at_ks(test_data=train_data,
                                              ks=cutoffs,
                                              label_type=LABEL_TYPE.MultiLabel,
                                              device='cpu')
            np_train_ndcg_ks = train_ndcg_ks.data.numpy()
            list_train_ndcgs.append(np_train_ndcg_ks)

        test_ndcgs = np.vstack(list_test_ndcgs)
        train_ndcgs = np.vstack(list_train_ndcgs)

        return list_losses, train_ndcgs, test_ndcgs

    def set_data_setting(self,
                         data_json=None,
                         debug=False,
                         data_id=None,
                         dir_data=None):
        if data_json is not None:
            self.data_setting = DataSetting(data_json=data_json)
        else:
            self.data_setting = DataSetting(debug=debug,
                                            data_id=data_id,
                                            dir_data=dir_data)

    def get_default_data_setting(self):
        return self.data_setting.default_setting()

    def iterate_data_setting(self):
        return self.data_setting.grid_search()

    def set_eval_setting(self, eval_json=None, debug=False, dir_output=None):
        if eval_json is not None:
            self.eval_setting = EvalSetting(debug=debug, eval_json=eval_json)
        else:
            self.eval_setting = EvalSetting(debug=debug, dir_output=dir_output)

    def get_default_eval_setting(self):
        return self.eval_setting.default_setting()

    def iterate_eval_setting(self):
        return self.eval_setting.grid_search()

    def set_scoring_function_setting(self,
                                     sf_json=None,
                                     debug=None,
                                     sf_id=None):
        if sf_json is not None:
            self.sf_parameter = ScoringFunctionParameter(sf_json=sf_json)
        else:
            self.sf_parameter = ScoringFunctionParameter(debug=debug,
                                                         sf_id=sf_id)

    def get_default_scoring_function_setting(self):
        return self.sf_parameter.default_para_dict()

    def iterate_scoring_function_setting(self):
        return self.sf_parameter.grid_search()

    def set_model_setting(self, model_id=None, dir_json=None, debug=False):
        """
        Initialize the parameter class for a specified model
        :param debug:
        :param model_id:
        :return:
        """
        if model_id in [
                'RankMSE', 'ListMLE', 'ListNet', 'RankCosine', 'DASALC',
                'HistogramAP'
        ]:  # ModelParameter is sufficient
            self.model_parameter = ModelParameter(model_id=model_id)
        else:
            if dir_json is not None:
                para_json = dir_json + model_id + "Parameter.json"
                self.model_parameter = globals()[model_id + "Parameter"](
                    para_json=para_json)
            else:  # the 3rd type, where debug-mode enables quick test
                self.model_parameter = globals()[model_id +
                                                 "Parameter"](debug=debug)

    def get_default_model_setting(self):
        return self.model_parameter.default_para_dict()

    def iterate_model_setting(self):
        return self.model_parameter.grid_search()

    def declare_global(self, model_id=None):
        """
        Declare global variants if required, such as for efficiency
        :param model_id:
        :return:
        """
        if model_id == 'WassRank':  # global buffering across a number of runs with different model parameters
            self.dict_cost_mats, self.dict_std_dists = dict(), dict()

    def point_run(self,
                  debug=False,
                  model_id=None,
                  sf_id=None,
                  data_id=None,
                  dir_data=None,
                  dir_output=None,
                  dir_json=None,
                  reproduce=False):
        """
        Perform one-time run based on given setting.
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """
        if dir_json is None:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug,
                                  data_id=data_id,
                                  dir_data=dir_data)
            self.set_scoring_function_setting(debug=debug, sf_id=sf_id)
            self.set_model_setting(debug=debug, model_id=model_id)
        else:
            data_eval_sf_json = dir_json + 'Data_Eval_ScoringFunction.json'
            self.set_eval_setting(eval_json=data_eval_sf_json)
            self.set_data_setting(data_json=data_eval_sf_json)
            self.set_scoring_function_setting(sf_json=data_eval_sf_json)
            self.set_model_setting(model_id=model_id, dir_json=dir_json)

        data_dict = self.get_default_data_setting()
        eval_dict = self.get_default_eval_setting()
        sf_para_dict = self.get_default_scoring_function_setting()
        model_para_dict = self.get_default_model_setting()

        self.declare_global(model_id=model_id)

        if reproduce:
            self.kfold_cv_reproduce(data_dict=data_dict,
                                    eval_dict=eval_dict,
                                    model_para_dict=model_para_dict,
                                    sf_para_dict=sf_para_dict)
        else:
            self.kfold_cv_eval(data_dict=data_dict,
                               eval_dict=eval_dict,
                               model_para_dict=model_para_dict,
                               sf_para_dict=sf_para_dict)

    def grid_run(self,
                 model_id=None,
                 sf_id=None,
                 dir_json=None,
                 debug=False,
                 data_id=None,
                 dir_data=None,
                 dir_output=None,
                 argobj=None):
        """
        Explore the effects of different hyper-parameters of a model based on grid-search
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """
        if dir_json is not None:
            data_eval_sf_json = dir_json + 'Data_Eval_ScoringFunction.json'
            self.set_data_setting(data_json=data_eval_sf_json)
            self.set_scoring_function_setting(sf_json=data_eval_sf_json)
            self.set_eval_setting(debug=debug, eval_json=data_eval_sf_json)
            self.set_model_setting(model_id=model_id, dir_json=dir_json)
        else:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug,
                                  data_id=data_id,
                                  dir_data=dir_data)
            self.set_scoring_function_setting(debug=debug, sf_id=sf_id)
            self.set_model_setting(debug=debug, model_id=model_id)

        self.declare_global(model_id=model_id)
        ''' select the best setting through grid search '''
        vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50]
        max_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
        k_index = cutoffs.index(vali_k)
        max_common_para_dict, max_sf_para_dict, max_model_para_dict = None, None, None

        for data_dict in self.iterate_data_setting():
            for eval_dict in self.iterate_eval_setting():
                assert self.eval_setting.check_consistence(
                    vali_k=vali_k, cutoffs=cutoffs)  # a necessary consistence

                for sf_para_dict in self.iterate_scoring_function_setting():
                    for model_para_dict in self.iterate_model_setting():
                        self.kfold_cv_eval(
                            data_dict=data_dict,
                            eval_dict=eval_dict,
                            sf_para_dict=sf_para_dict,
                            model_para_dict=model_para_dict,
                            argobj=argobj)


    def run(self,
            debug=False,
            model_id=None,
            sf_id=None,
            config_with_json=None,
            dir_json=None,
            data_id=None,
            dir_data=None,
            dir_output=None,
            grid_search=False,
            reproduce=False,
            argobj=None):
        if config_with_json:
            assert dir_json is not None
            if reproduce:
                self.point_run(debug=debug,
                               model_id=model_id,
                               dir_json=dir_json,
                               reproduce=reproduce)
            else:
                self.grid_run(debug=debug,
                              model_id=model_id,
                              dir_json=dir_json,
                              argobj=argobj)
        else:
            assert sf_id in ['pointsf', 'listsf']
            if grid_search:
                self.grid_run(debug=debug,
                              model_id=model_id,
                              sf_id=sf_id,
                              data_id=data_id,
                              dir_data=dir_data,
                              dir_output=dir_output)
            else:
                self.point_run(debug=debug,
                               model_id=model_id,
                               sf_id=sf_id,
                               data_id=data_id,
                               dir_data=dir_data,
                               dir_output=dir_output,
                               reproduce=reproduce)
