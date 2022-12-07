#!/usr/bin/env python
# coding: utf-8


import warnings, logging
import os, glob, json, gc
import getpass, random
import rasterio
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import catboost as cat
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import multiprocessing
import joblib
from joblib import Parallel, delayed
from scipy.stats import gmean
from time import sleep
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


if __name__ == '__main__':
    input_path = os.environ['INPUT_DATA']
    model_path = os.path.join(input_path, 'checkpoint/crossval-catboost/')
    output_path = os.environ['OUTPUT_DATA']

    test = pd.read_csv(f'{output_path}/Final_Test.csv')

    main_cols = test.columns.difference(['fid', 'crop_id', 'fold', 'target', 'label'], sort=False).tolist()

    n_split = 14
    model_list = [f'{model_path}crossval-catboost-{n}.cbm' for n in range(n_split)]

    oofs = []
    for model_name in tqdm(model_list):
        model = cat.CatBoostClassifier()
        model.load_model(model_name)

        oofs.append(
            model.predict_proba(test[main_cols])
        )
    predic = gmean(oofs, axis=0)


    agr = [
        'Wheat', 'Mustard','Lentil','No Crop','Green pea','Sugarcane','Garlic','Maize','Gram','Coriander','Potato','Bersem','Rice'
    ]

    submission = pd.DataFrame()
    submission['field_id'] = test['fid']

    for i, label in enumerate(agr):
        submission[label] = np.array(predic)[:,i]

    submission.to_csv(f'{output_path}/crossval-catboost.csv', index=False)
