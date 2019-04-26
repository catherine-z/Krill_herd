import NiaPy
from NiaPy.algorithms.basic.kh import KrillHerdV1, KrillHerdV2, KrillHerdV3, KrillHerdV4, KrillHerdV11
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.cluster
import numpy as np

import warnings
warnings.filterwarnings("ignore")

raw_data = list()
n = 100
path = '/Users/apple/Downloads/maildir/arnold-j/all_documents'
for filename in os.listdir(path)[:n]:
    with open(path+'/'+filename, 'rb') as f:
        contents = f.read()
        contents = contents.replace('\r\n', ' ')
        contents = contents.replace('\n', ' ')
        raw_data.append(contents)

cv = CountVectorizer(stop_words = 'english', token_pattern='[a-zA-Z]+')
cv_fit = cv.fit(raw_data)
cv_mat = cv.fit_transform(raw_data)
cv_array = np.array(cv_mat.todense())

length = np.sum(cv_mat.todense(),axis=1).astype(float)
frequence = np.array(cv_array/length)
doc_count = np.sum((cv_array>0)*1, axis=0)
weight = frequence * np.log(n/doc_count)

cv_df = pd.DataFrame(data=weight)
words = np.array(cv_fit.get_feature_names())
cv_df.columns = words

km = sklearn.cluster.KMeans(n_clusters=10)
km_fit = km.fit(cv_df)
km_pred = km.fit_predict(cv_df)