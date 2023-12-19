import os
output_path = ""
root_path = "."
kfolds = 5
for fold in range(1, kfolds+1):
    os.system(f'python {root_path}/DNN/train.py -d {root_path}/data/davidson/kfolds/train_{fold}.csv --trial {root_path}/data/davidson/kfolds/test_{fold}.csv -s {root_path}/data/sentiment_datasets/train_E6oV3lV.csv --word_list {root_path}/data/word_list/word_all.txt --emb {root_path}/data/glove.6B.300d.txt -o {root_path}/outputs/output_DV_{fold} -b 512 --epochs 30 --lr 0.002 --maxlen 50 -t HHMM_transformer')