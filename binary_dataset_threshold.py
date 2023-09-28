import numpy as np

root_path = ''
file_paths = [f'{root_path}/mslr/Fold1/', f'{root_path}/yahoo/ltrc_yahoo/', f'{root_path}/istella-s-letor/sample/']
datasets = ['mslr', 'set1', 'istella']

t = 4.0
tau = 4.5
for i, dataset in enumerate(datasets):
    splits = ['train', 'vali', 'test']
    if dataset == 'set1':
        splits = ['set1.train', 'set1.valid', 'set1.test']
    
    for split in splits:
        print(f'############\n\n{dataset}, {split}\n\n############')
        counter = 0
        positives = 0
        with open(file_paths[i] + f'{split}_t={t}_tau={tau}.txt', 'w+') as f:
            with open(file_paths[i] + f'{split}.txt', "r") as file:
                for line in file:
                    data_split = line.split(' ')
                    threshold_data_split = data_split.copy()
                    relevance = float(line.split(' ')[0])

                    g0 = np.random.gumbel()
                    g1 = np.random.gumbel()
                    lhs = t * relevance + g1
                    rhs = t * tau + g0
                    if lhs > rhs:
                        threshold_data_split[0] = '1'
                        positives += 1
                    else:
                        threshold_data_split[0] = '0'

                    threshold_line = ' '.join(threshold_data_split)
                    f.write(threshold_line)
                    counter += 1
                    if counter % 500000 == 0:
                        print(counter, 'datapoints processed')
        print('Fraction of positives', positives/counter)




