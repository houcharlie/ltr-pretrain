import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
import time

from itertools import product
from ptranking.base.utils import get_stacked_FFNet, get_resnet, ResNetBlock, LTRBatchNorm
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.pretrain.augmentations import zeroes, qgswap, gaussian, dacl, scarf
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from ptranking.metric.metric_utils import get_delta_ndcg
from torch.nn.init import xavier_normal_ as nr_init
from absl import logging
from torch.nn.init import eye_ as eye_init
from torch.nn.init import zeros_ as zero_init
class SimCLR(NeuralRanker):

    def __init__(self, id='SimCLRPretrainer', sf_para_dict=None, model_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(SimCLR, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.aug_percent = model_para_dict['aug_percent']
        self.dim = model_para_dict['dim']
        self.aug_type = model_para_dict['aug_type']
        self.temperature = model_para_dict['temp']
        self.mix = model_para_dict['mix']
        if self.aug_type == 'zeroes':
            self.augmentation = zeroes
        elif self.aug_type == 'qg':
            self.augmentation = qgswap
        elif self.aug_type == 'gaussian':
            self.augmentation = gaussian
        elif self.aug_type == 'scarf':
            self.augmentation = scarf
        elif self.aug_type == 'dacl':
            self.augmentation = dacl

    def init(self):
        self.point_sf = self.config_point_neural_scoring_function()
        # nr_hn = nn.Linear(136, 1)
        # self.point_sf.add_module('_'.join(['ff', 'scoring']), nr_hn)
        self.point_sf.to(self.device)
        self.projector = self.config_head()
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.loss_no_reduction = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        print(self.point_sf, file=sys.stderr)
        print(self.projector, file=sys.stderr)
        self.config_optimizer()


    def config_point_neural_scoring_function(self):
        point_sf = self.ini_pointsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: point_sf = point_sf.to(self.device)
        return point_sf

    def config_head(self):
        dim = self.dim
        prev_dim = -1
        for name, param in self.point_sf.named_parameters():
            if 'ff' in name and 'bias' not in name:
                prev_dim = param.shape[0]
        projector = nn.Sequential()

        for i in range(3):
            nr = nn.Linear(prev_dim, prev_dim)
            projector.add_module('_'.join(['project', 'linear', str(i)]), nr)
            projector.add_module('_'.join(['project', 'relu', str(i)]), nn.ReLU())

        # nr_block = ResNetBlock(prev_dim)
        # projector.add_module('_'.join(['projresnet', str(0)]), nr_block)
        # projector.add_module('_'.join(['project_bn']), LTRBatchNorm(prev_dim, momentum=0.1, affine=True, track_running_stats=False))
        # projector.add_module('_'.join(['project_act']), nn.ReLU())
        # for i in range(1):
        #     nr = nn.Linear(prev_dim, prev_dim)
        #     projector.add_module('_'.join(['project', 'linear', str(i)]), nr)
        #     projector.add_module('_'.join(['project', 'relu', str(i)]), nn.ReLU())
        nr = nn.Linear(prev_dim, dim)
        projector.add_module('_'.join(['project', 'linear', 'final']), nr)
        if self.gpu: projector = projector.to(self.device)

        return projector

    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.projector.parameters():
            all_params.append(param)
        # for param in self.scorer.parameters():
        #     all_params.append(param)
        
        return nn.ParameterList(all_params)
        # return self.point_sf.parameters()

    def ini_pointsf(self, num_features=None, h_dim=100, out_dim=136, num_layers=3, AF='R', TL_AF='S', apply_tl_af=False,
                    BN=True, bn_type=None, bn_affine=False, dropout=0.1):
        '''
        Initialization of a feed-forward neural network
        '''
        encoder_layers = num_layers
        ff_dims = [num_features]
        for i in range(encoder_layers):
            ff_dims.append(h_dim)
        ff_dims.append(out_dim)

        # point_sf = get_stacked_FFNet(ff_dims=ff_dims, AF=AF, TL_AF=TL_AF, apply_tl_af=apply_tl_af, dropout=dropout,
        #                              BN=BN, bn_type=bn_type, bn_affine=bn_affine, device=self.device)
        h_dim = 136
        point_sf = get_resnet(num_features, h_dim)
        return point_sf

    def info_nce_loss(self, features, batch_size):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1).to(self.device)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1).to(self.device)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1).to(self.device)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1).to(self.device)

        logits = torch.cat([positives, negatives], dim=1).to(self.device)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature

        return logits, labels

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        x1 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device)
        x2 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device)

        data_dim = batch_q_doc_vectors.shape[2]
        # x1_flat = x1.reshape((-1, data_dim))
        # x2_flat = x2.reshape((-1, data_dim))
        embed1 = self.point_sf(x1)
        embed2 = self.point_sf(x2)
        z1 = self.projector(embed1)
        z2 = self.projector(embed2)
        
        s1 = z1.view(-1, self.dim)  # [batch_size, embed_dim]
        s2 = z2.view(-1, self.dim)  # [batch_size, embed_dim]


        s_concat = torch.cat((s1, s2), dim=0).to(self.device)
        logits_qg, labels_qg = self.info_nce_loss(s_concat, s1.shape[0])
        return logits_qg, labels_qg
    
    def sub_forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        x1 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device)
        x2 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device)

        data_dim = batch_q_doc_vectors.shape[2]
        # x1_flat = x1.reshape((-1, data_dim))
        # x2_flat = x2.reshape((-1, data_dim))
        embed1 = self.point_sf(x1)
        embed2 = self.point_sf(x2)
        z1 = self.projector(embed1)
        z2 = self.projector(embed2)
        
        s1 = z1.view(-1, self.dim)  # [batch_size, embed_dim]
        s2 = z2.view(-1, self.dim)  # [batch_size, embed_dim]

        # shuffle all the elements randomly
        randidx = torch.randperm(s1.shape[0])
        b1 = s1[randidx, :]
        b2 = s2[randidx, :]
        # new "query groups"
        b1 = b1.reshape(-1, num_docs, self.dim)
        b2 = b2.reshape(-1, num_docs, self.dim)  

        s_concat = torch.cat((b1, b2), dim=1).to(self.device)
        logits_qg, labels_qg = self.qg_info_nce_loss(s_concat, z1.shape[1], z1.shape[0])
        return logits_qg, labels_qg

    def qg_forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        x1 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device)
        x2 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device)

        data_dim = batch_q_doc_vectors.shape[2]
        # x1_flat = x1.reshape((-1, data_dim))
        # x2_flat = x2.reshape((-1, data_dim))
        embed1 = self.point_sf(x1)
        embed2 = self.point_sf(x2)
        z1 = self.projector(embed1)
        z2 = self.projector(embed2)


        z_concat = torch.cat((z1, z2), dim=1).to(self.device)
        logits_qg, labels_qg = self.qg_info_nce_loss(z_concat, z1.shape[1], z1.shape[0])
        return logits_qg, labels_qg

    def qg_info_nce_loss(self, features, qg_size, batch_size):
        # features: [batchsize, 2 x qgsize, embed_dim]

        # [2 x qgsize]
        labels = torch.cat([torch.arange(qg_size) for i in range(2)], dim=0)
        # [2 x qgsize, 2 x qgsize] 4 identity matrices in each quadrant
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        # [batchsize, 2 x qgsize, 2 x qgsize]
        labels = labels[None, :, :].expand(batch_size, -1, -1)
        # [batchsize, 2 x qgsize, embed_dim]
        features = F.normalize(features, dim=2)
        # [batchsize, 2 x qgsize, embed_dim] x [batchsize, embed_dim, 2 x qgsize] = [batchsize, 2 x qgsize, 2 x qgsize] 
        similarity_matrix = torch.bmm(features, features.permute(0, 2, 1))
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[1], dtype=torch.bool).to(self.device)[None, :, ].expand(batch_size, -1, -1)
        labels = labels[~mask].view(batch_size, labels.shape[1], -1).to(self.device)
        similarity_matrix = similarity_matrix[~mask].view(batch_size, similarity_matrix.shape[1], -1).to(self.device)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        # [batchsize, 2 x qgsize, 1]
        positives = similarity_matrix[labels.bool()].view(batch_size, labels.shape[1], -1).to(self.device)
        # select only the negatives the negatives
        # [batchsize, 2 x qgsize, 2 x qgsize - 2]
        negatives = similarity_matrix[~labels.bool()].view(batch_size, similarity_matrix.shape[1], -1).to(self.device)
        # [batchsize, 2 x qgsize, 2 x qgsize - 1]
        logits = torch.cat([positives, negatives], dim=2).to(self.device)
        # [batchsize, 2 x qgsize]
        labels = torch.zeros(logits.shape[1], dtype=torch.long).to(self.device)[None, :].expand(batch_size, -1)
        # [batchsize, 2 x qgsize, 2 x qgsize - 1]
        logits = logits / self.temperature

        return logits, labels


    def train(self, train_data, epoch_k=None, **kwargs):
        '''
        One epoch training using the entire training data
        '''
        self.train_mode()

        assert 'label_type' in kwargs and 'presort' in kwargs
        label_type, presort = kwargs['label_type'], kwargs['presort']
        num_queries = 0
        epoch_loss = torch.tensor([0.0], device=self.device)
        batches_processed = 0
        all_correct = torch.tensor([0.0], device=self.device)
        all_attempts = torch.tensor([0.0], device=self.device)
        start_time = time.time()

        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data: # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            (batch_loss, correct, attempts), stop_training = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k, presort=presort, label_type=label_type)
            
            if stop_training:
                break
            else:
                epoch_loss += batch_loss.item()
                all_correct += correct
                all_attempts += attempts
            batches_processed += 1
        print("---One epoch time %s seconds ---" % (time.time() - start_time), file=sys.stderr)
            # print(batches_processed, file=sys.stderr)
        total_norm = 0.
        for p in self.point_sf.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print('Curr norm', total_norm, file=sys.stderr)
        print('Epoch accuracy', all_correct/all_attempts, 'out of', all_attempts, file=sys.stderr)
        epoch_loss = epoch_loss/num_queries
        return epoch_loss, stop_training

    def eval_mode(self):
        self.point_sf.eval()
        self.projector.eval()

    def train_mode(self):
        self.point_sf.train(mode=True)
        self.projector.train(mode=True)

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.point_sf.state_dict(), dir + name)
        torch.save(self.projector.state_dict(), dir + name + 'projector')

    def load(self, file_model, **kwargs):
        device = kwargs['device']
        self.point_sf.load_state_dict(torch.load(file_model, map_location=device))
        self.projector.load_state_dict(torch.load(file_model + 'projector', map_location=device))

    def get_tl_af(self):
        return self.sf_para_dict[self.sf_para_dict['sf_id']]['TL_AF']

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch_size, num_docs, num_features]
        @param batch_std_labels: not used
        @param kwargs:
        @return:
        '''
        
        logits_qg, labels_qg = batch_preds
        ### for SimCLR ###
        lambda_loss = self.loss(logits_qg, labels_qg)
        pred = torch.argmax(logits_qg, dim=1)
        correct = torch.sum(pred == labels_qg)
        total_num = pred.shape[0]
        ### for SimCLR ###

        ### for simclr_rank ###
        # loss = self.loss_no_reduction(logits_qg.permute(0, 2, 1), labels_qg)
        # loss_reduced = loss.mean(dim = 1)
        # lambda_loss = loss_reduced.mean()

        # # [batchsize, 2 x qgsize]
        # pred = torch.argmax(logits_qg, dim=2)
        # # [batchsize, 2 x qgsize]
        # correct = torch.sum(pred == labels_qg)
        # total_num = pred.shape[0] * pred.shape[1]
        ### for simclr_rank ###

        loss = lambda_loss
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.point_sf.parameters(), 2.0)
        self.optimizer.step()
        return loss, correct, total_num
    
    def train_op(self, batch_q_doc_vectors, batch_std_labels, **kwargs):
        '''
        The training operation over a batch of queries.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance labels for documents associated with the same query.
        @param kwargs: optional arguments
        @return:
        '''
        stop_training = False
        batch_preds = self.forward(batch_q_doc_vectors)

        return self.custom_loss_function(batch_preds, batch_std_labels, **kwargs), stop_training
    
    def validation(self, vali_data=None, vali_metric=None, k=5, presort=False, max_label=None, label_type=LABEL_TYPE.MultiLabel, device='cpu'):
        self.eval_mode() # switch evaluation mode

        num_queries = 0
        sum_val_loss = torch.zeros(1).to(self.device)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in vali_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if batch_std_labels.size(1) < k:
                continue  # skip if the number of documents is smaller than k
            else:
                num_queries += len(batch_ids)

            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            val_loss = self.custom_loss_function(batch_preds, batch_std_labels)

            sum_val_loss += val_loss # due to batch processing

        avg_val_loss = val_loss / num_queries
        return avg_val_loss.cpu()

    def adhoc_performance_at_ks(self, test_data=None, ks=[1, 5, 10], label_type=LABEL_TYPE.MultiLabel, max_label=None,
                                presort=False, device='cpu', need_per_q=False):
        '''
        Compute the performance using multiple metrics
        '''
        self.eval_mode()  # switch evaluation mode

        val_loss = self.validation(test_data)
        output = torch.zeros(len(ks))
        output[0] = val_loss
        output[1] = val_loss
        return output, output, output, output, output


class SimCLRParameter(ModelParameter):
    def __init__(self, debug=False, para_json=None):
        super(SimCLRParameter, self).__init__(model_id='SimCLR', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for SimRank
        :return:
        """
        self.para_dict = dict(model_id=self.model_id, aug_percent=0.7, dim=100, aug_type='qg', temp=0.07, mix=0.5)
        return self.para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        para_dict = given_para_dict if given_para_dict is not None else self.para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')
        para_str = s1.join(['aug_percent', '{:,g}'.format(para_dict['aug_percent']), 'embed_dim', '{:,g}'.format(para_dict['dim']), 'aug_type', para_dict['aug_type'], 'temp', para_dict['temp'], 'mix', para_dict['mix']])
        return para_str

    def grid_search(self):
        """
        Iterator of parameter settings for simrank
        """
        if self.use_json:
            choice_aug = self.json_dict['aug_percent']
            choice_dim = self.json_dict['dim']
            choice_augtype = self.json_dict['aug_type']
            choice_temp = self.json_dict['temp']
            choice_mix = self.json_dict['mix']
        else:
            choice_aug = [0.3, 0.7] if self.debug else [0.7]  # 1.0, 10.0, 50.0, 100.0
            choice_dim = [50, 100] if self.debug else [100]  # 1.0, 10.0, 50.0, 100.0
            choice_augtype = ['zeroes', 'qg'] if self.debug else ['qg']  # 1.0, 10.0, 50.0, 100.0
            choice_temp = [0.07, 0.1] if self.debug else [0.07] 
            choice_mix = [1., 0.] if self.debug else [1.]


        for aug_percent, dim, augtype, temp, mix in product(choice_aug, choice_dim, choice_augtype, choice_temp, choice_mix):
            self.para_dict = dict(model_id=self.model_id, aug_percent=aug_percent, dim=dim, aug_type=augtype, temp=temp, mix=mix)
            yield self.para_dict