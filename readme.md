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
After download the data and the pre-trained word vectors, just run the sample_run.sh

