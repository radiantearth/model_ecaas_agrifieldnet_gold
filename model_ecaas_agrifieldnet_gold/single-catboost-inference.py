#!/usr/bin/env python
# coding: utf-8

import catboost
import os, random
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge ,LinearRegression, LogisticRegression
from sklearn.model_selection import KFold,StratifiedKFold ,GroupKFold, cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.cluster import KMeans
import warnings
from scipy.stats import gmean
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    input_path = os.environ['INPUT_DATA']
    model_path = os.path.join(input_path, 'checkpoint/single-catboost/')
    output_path = os.environ['OUTPUT_DATA']


    test = pd.read_csv(f'{output_path}/Final_Test.csv')

    drop_cols = ['fid', 'label', 'target','lat3', 'rot45_y', 'evi2_max', 'field_tile_size', 'rot45_x', 'evi2_median','evi_max','S2REP_max']
    use_columns = test.columns.difference(drop_cols).tolist()

    tes = test[use_columns]


    agr = [
        'Wheat', 'Mustard','Lentil','No Crop','Green pea','Sugarcane','Garlic','Maize','Gram','Coriander','Potato','Bersem','Rice'
    ]

    oofs = []
    n_models = 15
    model_list = [f'{model_path}single-catboost-{n}.cbm' for n in range(n_models)]

    for model_name in tqdm(model_list):
        model = CatBoostClassifier()
        model.load_model(model_name)

        oofs.append(
            model.predict_proba(tes)
        )

    predic = gmean(oofs, axis=0)


    submission = pd.DataFrame()
    submission['field_id'] = test['fid']
    for i, label in enumerate(agr):
        submission[label] = np.array(predic)[:,i]

    submission.to_csv(f'{output_path}/single-catboost.csv', index=False)





