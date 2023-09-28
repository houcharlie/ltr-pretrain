# How to run the code

Step 1: You must first download the datasets and extract them.  They
are located in the following locations:

https://www.microsoft.com/en-us/research/project/mslr/

https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&guccounter=1

https://istella.ai/data/letor-dataset/

Step 2: You must fill in the dataset location you want to train on in the 
config files located in ssl_rank/inputs/lambdarank/Data_Eval_ScoringFunction.json for lambdarank for example (or ssl_rank/testing/ltr_tree/json/Tree_Data_Eval_ScoringFunction.json for GBDTs)
as well as the data id (MSLRWEB30K, Set1, Istella_S).  You can also fill in the approximate
batch size in "tr_batch_size" and the number of epochs to train in "epochs".  

Step 3: Run the command to pretrain and finetune a method.  

An example input for pretraining with SimCLR-Rank (can also use SimSiam) and finetuning with LambdaRank when there are 0.001 of the training query groups remaining:


>python e2e_eval.py -cuda 0 -dir_json ssl_rank/inputs -pretrain_lr 0.001 -finetune_lr 0.001 -trial_num 0 -aug_type zeroes -aug_percent 0.1 -dim 64 -layers 5 -pretrainer SimCLR -shrink 0.001 -freeze 0 -probe_layers 1 -finetune_only 0 -finetune_trials 0


An example input for running a GBDT baseline with 0.001 of the training query groups remaining:

>python e2e_eval.py -cuda 0 -dir_json ssl_rank/testing/ltr_tree/json -pretrain_lr 0.001 -finetune_lr 0.001 -trial_num 0 -aug_type zeroes -aug_percent 0.1 -dim 64 -layers 5 -pretrainer LightGBMLambdaMART -shrink 0.001 -freeze 0 -probe_layers 1 -finetune_only 0 -finetune_trials 0


An example input for running a deep learning baseline with 0.001 of the training query groups remaining:

>python e2e_eval.py -cuda 0 -dir_json ssl_rank/inputs -pretrain_lr 0.001 -finetune_lr 0.001 -trial_num 0 -aug_type none -aug_percent 0.1 -dim 64 -layers 5 -pretrainer SimCLR -shrink 0.001 -freeze 0 -probe_layers 1 -finetune_only 0 -finetune_trials 0


# How to run sparse training set experiments
One can do this by running binary_dataset_threshold.py then binary_dataset_filter.py (modify the script to 
produce the dataset of appropriate sparsity) and then modifying the config files' data paths.

# Necessary packages
Check requirements.txt for the packages needed to run the scripts.

