<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS117 - Xử lý ngôn ngữ tự nhiên</b></h1>



## BẢNG MỤC LỤC
* [ Giảng viên hướng dẫn](#giangvien)
* [ Thành viên nhóm](#thanhvien)
* [ Đồ án môn học](#doan)
## GIẢNG VIÊN HƯỚNG DẪN
<a name="giangvien"></a>
* TS. **Nguyễn Thị Quý**

## THÀNH VIÊN NHÓM
<a name="thanhvien"></a>
| STT    | MSSV          | Họ và Tên               | Email                   |
| ------ |:-------------:| ----------------------:|-------------------------:
| 1      | 21522110      | Bùi Mạnh Hùng          |21522110@gm.uit.edu.vn   |
| 2      | 21522295      | Lê Trần Bảo Lợi          |21522295@gm.uit.edu.vn   |
| 3      | 21522487      | Bùi Đình Quân         |21522487@gm.uit.edu.vn   |
| 4      | 21522621      | Huỳnh Công Thiện          |21522621@gm.uit.edu.vn   |
| 5      | 21522706      | Nguyễn Minh Trí           |21522706@gm.uit.edu.vn   |
## ĐỒ ÁN MÔN HỌC
<a name="doan"></a>
## Demo
# SKS
This repository provides code for the paper "Hate Speech Detection based on Sentiment Knowledge Sharing"

![avatar](figure1.jpg)

# Requirements
Python = 3.9
# run on colab
+ !python -m nltk.downloader 'punkt'
+ !sudo apt-get update
+ !sudo apt-get install python3-enchant -y

## Data
- ## **_Download models here !!! [MODELS](https://drive.google.com/drive/folders/1NMphGjxDkNpc1j8z6JY7_sgrS0yeoO72?usp=sharing)_**
- ## **_Don't care about the data below; just download it here!!! [DATA-SKS](https://drive.google.com/file/d/1XmpvGpbPpjUn4MtuhMvxRetLLk_vZne4/view?usp=sharing)_**

- ### SemEval data-set

    We provide the trainig- and test-set for the [SemEval2019 data-set](http://hatespeech.di.unito.it/hateval.html) as two separate csv files `df_train.csv` and `df_test.csv`. To accomodate the original implementation, the original fields `id`, `text` and `HS` have already been renamed as `task_idx`, `tweet` and `label`.

- ### Davidson data-set

    We include both the original [Davidson data-set](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data) `davidson_data_full.csv` and our 5-fold cross-validation splits, where the `class` field has already been renamed as `label` to accomodate the original implementation.

- ### Sentiment data-setr

    We provide the training data-set used for the sentiment analysis task `train_E6oV3lV.csv`. The original training- and test-set are freely available on [Kaggle](https://www.kaggle.com/dv1453/twitter-sentiment-analysis-analytics-vidya).

- ### Dictionary of derogatory words

    We rely on the same dictionary of derogatory words `word_all.txt` compiled by the original authors.

The SE dataset may need some adjustment in formatting from tsv to csv. Make sure to put these in the data directory and also within their respective directory too. ex: `SemEval_task5/df_test.csv`

The glove txt file can be downloaded [here](https://www.kaggle.com/datasets/aellatif/glove6b300dtxt). There is also a larger one available, but make sure to adjust the script for it [here](https://www.kaggle.com/datasets/authman/pickled-glove840b300d-for-10sec-loading)
# Usage
## **Train**
`python train.py -d path_data_train --trial path_data_test -s ..\train_E6oV3lV.csv --word_list  ..\word_all.txt --emb ..\glove.6B.300d.txt -o ..\output_dir -b 512 --epochs 60 --lr 0.002 --maxlen 50 -t HHMM_transformer
`
## **Test**
`python inference.py --word_list ..\word_all.txt  --maxlen 50 --vocab-path ..\vocab.pkl`