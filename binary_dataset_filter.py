import os
import numpy as np

t = 4.0
tau = 4.5
root_path = ''
file_paths = [f'{root_path}/mslr/Fold1/', f'{root_path}/yahoo/ltrc_yahoo/', f'{root_path}/istella-s-letor/sample/']
datasets = ['mslr', 'set1', 'istella']
new_paths = [f'{root_path}/mslr/shift_sparse_t={t}_tau={tau}/Fold1/', f'{root_path}/yahoo/ltrc_yahoo/shift_sparse_t={t}_tau={tau}/', f'{root_path}/istella-s-letor/sample/shift_sparse_t={t}_tau={tau}/']


for i, dataset in enumerate(datasets):
    splits = ['train', 'vali', 'test']
    if dataset == 'set1':
        splits = ['set1.train', 'set1.valid', 'set1.test']
    for q, split in enumerate(splits):
        print(f'############\n\n{dataset}, {split}\n\n############')

        if q > 1:
            with open(new_paths[i] + f'{split}.txt', "w+") as f:
                with open(file_paths[i] + f'{split}.txt', 'r') as file:
                    for line in file:
                        f.write(line)
        else:
            counter = 0
            valid_qgs = 0
            total_qgs = 0
            if not os.path.exists(new_paths[i]):
                os.makedirs(new_paths[i])
            with open(new_paths[i] + f'{split}.txt', "w+") as f:
                with open(file_paths[i] + f'{split}_t={t}_tau={tau}.txt', 'r') as file:
                    qg_positives_ratio_sum = 0
                    curr_qg_id = None
                    curr_qg = []
                    has_click = False
                    items_in_qg = 0
                    positives_in_qg = 0
                    for line in file:
                        data_split = line.split(' ')
                        qg_id = data_split[1].split(':')[1]
                        relevance = data_split[0]

                        if qg_id != curr_qg_id:
                            total_qgs += 1
                            # unload the buffer of qg items
                            if len(curr_qg) > 0 and has_click:
                                valid_qgs += 1
                                for asin in curr_qg:
                                    f.write(asin)
                                qg_positives_ratio_sum += positives_in_qg / items_in_qg
                            # reset curr_qg
                            curr_qg = []
                            # we are on a new qg
                            curr_qg_id = qg_id
                            has_click = False

                            positives_in_qg = 0
                            items_in_qg = 0
                        if relevance == '1':
                            positives_in_qg += 1
                        items_in_qg += 1
                        has_click = has_click or (relevance == '1')
                        curr_qg.append(line)
                        counter += 1
                        if counter % 500000 == 0:
                            print(counter, 'datapoints processed')
                        
                    # last qg
                    total_qgs += 1
                    if len(curr_qg) > 0 and has_click:
                        valid_qgs += 1
                        for asin in curr_qg:
                            f.write(asin)
                        if items_in_qg > 0:
                            qg_positives_ratio_sum += positives_in_qg / items_in_qg
                    # reset curr_qg
                    curr_qg = []
                    # we are on a new qg
                    curr_qg_id = qg_id
                    has_click = False
                    print('valid qgs ratio', valid_qgs / total_qgs)
                    print('in-qg positives ratio', qg_positives_ratio_sum / valid_qgs)
