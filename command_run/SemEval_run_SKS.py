import os
output_path = ""
root_path = ""
os.system(f"python {root_path}/DNN/train.py -d {root_path}/data/SemEval_task5/df_train.csv --trial {root_path}/data/SemEval_task5/df_test.csv -s {root_path}/data/sentiment_datasets/train_E6oV3lV.csv --word_list {root_path}/data/word_list/word_all.txt --emb {root_path}/data/glove.840B.300d.txt -o {root_path}/outputs/SemEval -b 512 --epochs 30 --lr 0.002 --maxlen 50 -t HHMM_transformer")