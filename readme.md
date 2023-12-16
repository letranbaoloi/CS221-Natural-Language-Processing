# SKS
This repository provides code for the paper "Hate Speech Detection based on Sentiment Knowledge Sharing"

![avatar](figure1.jpg)

# Requirements
Python = 3.9

# Prepare data
+ [DV](https://github.com/t-davidson/hate-speech-and-offensive-language)
+ [SE](https://github.com/rnjtsh/hatEval-2019/blob/master/public_development_en/dev_en.tsv)
+ [SA](https://www.kaggle.com/dv1453/twitter-sentiment-analysis-analytics-vidya)

The SE dataset may need some adjustment in formatting from tsv to csv. Make sure to put these in the data directory and also within their respective directory too. ex: `SemEval_task5/df_test.csv`

The glove txt file can be downloaded [here](https://www.kaggle.com/datasets/aellatif/glove6b300dtxt). There is also a larger one available, but make sure to adjust the script for it [here](https://www.kaggle.com/datasets/authman/pickled-glove840b300d-for-10sec-loading)
# Usage
After download the data and the pre-trained word vectors, just run the sample_run.sh

