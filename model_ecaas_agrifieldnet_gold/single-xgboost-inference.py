#!/usr/bin/env python
# coding: utf-8


import os
import xgboost
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import Ridge ,LinearRegression, LogisticRegression
from sklearn.model_selection import KFold,StratifiedKFold ,GroupKFold, cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import gmean
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    input_path = os.environ['INPUT_DATA']
    model_path = os.path.join(input_path, 'checkpoint/single-xgboost/')
    output_path = os.environ['OUTPUT_DATA']


    test = pd.read_csv(f'{output_path}/Final_Test.csv')


    drop_cols = ['fid', 'label','lat3', 'rot45_y', 'evi2_max', 'field_tile_size', 'rot45_x', 'evi2_median']
    use_columns = test.columns.difference(drop_cols).tolist()

    tes = test[use_columns]

    agr = [
        'Wheat', 'Mustard','Lentil','No Crop','Green pea','Sugarcane','Garlic','Maize','Gram','Coriander','Potato','Bersem','Rice'
    ]

    oofs = []
    n_models = 15
    model_list = [f'{model_path}single-xgboost-{n}.json' for n in range(n_models)]

    for model_name in tqdm(model_list):
        model = XGBClassifier()
        model.load_model(model_name)

        oofs.append(
            model.predict_proba(tes[model.get_booster().feature_names])
        )

    predic = gmean(oofs, axis=0)

    submission = pd.DataFrame()
    submission['field_id'] = test['fid']

    for i, label in enumerate(agr):
        submission[label] = np.array(predic)[:,i]

    submission.to_csv(f'{output_path}/single-xgboost.csv', index=False)





