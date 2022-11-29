#!/usr/bin/env python
# coding: utf-8


import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import os, glob, json
import getpass, random
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import multiprocessing
import joblib
from joblib import Parallel, delayed
from scipy.stats import gmean
import pickle

def seed_all(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
        
    return seed


if __name__ == '__main__':

    seed_all()

    PATH = os.environ['INPUT_DATA']
    model_path = os.path.join(PATH, 'checkpoint/pixelwise-lightgbm/')
    output_path = os.environ['OUTPUT_DATA']

    main = 'ref_agrifieldnet_competition_v1'
    assets = ['field_ids','raster_labels']

    source_collection = f'{main}_source'
    train_label_collection = f'{main}_labels_train'
    test_label_collection = f'{main}_labels_test'


    ########################################################################################
    ############################ Utilities  ################################################
    ########################################################################################

    def get_folder_ids(label_collection):
      with open (f'{PATH}/{main}/{label_collection}/collection.json') as f:
          train_json = json.load(f)

      folder_ids = [i['href'].split('_')[-1].split('.')[0] for i in train_json['links'][4:]]

      return folder_ids

    def build_collection_df(label_collection):
      folder_ids = get_folder_ids(label_collection)

      data = [
          {
            'unique_folder_id': i,
            'field_paths': f'{PATH}/{main}/{label_collection}/{label_collection}_{i}/field_ids.tif',
            'label_paths': f'{PATH}/{main}/{label_collection}/{label_collection}_{i}/raster_labels.tif'
          }

          for i in folder_ids
      ]

      competition_data = pd.DataFrame(data)

      return competition_data


    def field_crop_extractor_v0(crop_field_files, fn):
        field_crops = {}

        for label_field_file in tqdm(crop_field_files):
            with rasterio.open(f'{PATH}/{main}/{train_label_collection}/{train_label_collection}_{label_field_file}/field_ids.tif') as src:
                field_data = src.read()[0]
            with rasterio.open(f'{PATH}/{main}/{train_label_collection}/{train_label_collection}_{label_field_file}/raster_labels.tif') as src:
                crop_data = src.read()[0]

            for x in range(0, crop_data.shape[0]):
                for y in range(0, crop_data.shape[1]):
                    field_id = str(field_data[x][y])
                    field_crop = crop_data[x][y]

                    if field_crops.get(field_id) is None:
                        field_crops[field_id] = []

                    if field_crop not in field_crops[field_id]:
                        field_crops[field_id].append(field_crop)

        field_crop_map  = [[k, v[0]]  for k, v in field_crops.items() ]
        field_crop = pd.DataFrame(field_crop_map , columns=['field_id','crop_id'])
        field_crop = field_crop[field_crop['field_id']!='0'].reset_index(drop=True)

        return field_crop

    def extract_field_crop_data(data):
      field_ids = []
      crop_type = []

      for i in tqdm(range(len(data))):
          with rasterio.open(data['field_paths'].iloc[i]) as src:
              field_data = src.read()[0]
          with rasterio.open(data['label_paths'].iloc[i]) as src:
              crop_data = src.read()[0]

          for field_id in np.unique(field_data)[1:]:
              ind = np.where(field_data == field_id)
              field_ids.append(field_id)
              crop_type.append(np.unique(crop_data[ind])[-1])

      df = pd.DataFrame(np.array([field_ids, crop_type]).transpose(),
                        columns=['field_id', 'crop_id'])
      return df

    def paralellize(fct, data, size, verbose=0, with_tqdm=False):
        fn = map(delayed(fct), data)
        if with_tqdm:
            fn = tqdm(fn, total=size)
        return Parallel(n_jobs=-1, verbose=verbose, backend="multiprocessing")(fn)

    def feature_extractor(data_ , path, fn):
        '''
            data_: Dataframe with 'field_paths' and 'unique_folder_id' columns
            path: Path to source collections files

            returns: pixel dataframe with corresponding field_ids
            '''

        X_arrays = []

        field_ids = np.empty((0, 1))

        for idx, tile_id in enumerate(tqdm(data_['unique_folder_id'])):

            field_src = rasterio.open( data_['field_paths'].values[idx])
            field_array = field_src.read(1)
            field_ids = np.append(field_ids, field_array.flatten())

            bands_src = [rasterio.open(f'{PATH}/{main}/{path}/{path}_{tile_id}/{band}.tif') for band in selected_bands]
            bands_array = [np.expand_dims(band.read(1).flatten(), axis=1) for band in bands_src]

            X_tile = np.hstack(bands_array)
            X_arrays.append(X_tile)

        X = np.concatenate(X_arrays)

        data = pd.DataFrame(X, columns=selected_bands)
        data['field_id'] = field_ids
        data = data[['field_id'] + selected_bands]
        data = data[data['field_id']!=0].reset_index(drop=True)

        return data

    def tile_feature_extractor(args):
      tile_id, fpath = args

      field_src = rasterio.open(fpath)
      field_array = field_src.read(1)
      field_ids = field_array.flatten()

      bands_src = [rasterio.open(f'{PATH}/{main}/{source_collection}/{source_collection}_{tile_id}/{band}.tif') for band in selected_bands]
      bands_array = [np.expand_dims(band.read(1).flatten(), axis=1) for band in bands_src]

      X_tile = np.hstack(bands_array)

      return X_tile, field_ids

    def tiles_to_df(tiles):
      X_arrays, field_ids = [], []

      for x, fids in tqdm(tiles):
        X_arrays.append(x)
        field_ids.append(fids)

      X = np.concatenate(X_arrays)
      field_ids = np.hstack(field_ids)

      data = pd.DataFrame(X, columns=selected_bands)
      data['field_id'] = field_ids
      data = data[['field_id'] + selected_bands]
      data = data[data['field_id']!=0].reset_index(drop=True)

      return data

    def paralellize(fct, data, size, verbose=0, with_tqdm=False):
        fn = map(delayed(fct), data)
        if with_tqdm:
            fn = tqdm(fn, total=size)
        return Parallel(n_jobs=-1, verbose=verbose, backend="multiprocessing")(fn)

    def vegetation_index():
      b01 = comb.filter(like = "B01").values[:,0]
      b02 = comb.filter(like = "B02").values[:,0]
      b03 = comb.filter(like = "B03").values[:,0]
      b04 = comb.filter(like = "B04").values[:,0]
      b05 = comb.filter(like = "B05").values[:,0]
      b06 = comb.filter(like = "B06").values[:,0]
      b07 = comb.filter(like = "B07").values[:,0]
      b08 = comb.filter(like = "B08").values[:,0]
      b8a = comb.filter(like = "B8A").values[:,0]
      b09 = comb.filter(like = "B09").values[:,0]
      b11 = comb.filter(like = "B11").values[:,0]
      b12 = comb.filter(like = "B12").values[:,0]

      eps = 0

      NDRE = (b08 - b06)/(b08 + b06 + eps)
      MMSR = ((b08/b04) - 1) / ((b08/b04)**0.5 + 1)
      NDWI = (b8a - b11)/(b8a + b11 + eps)
      GNDVI = (b08 - b03) / (b08 + b03 + eps)
      EVI2 = 2.5 * (b08 - b04) / (b08 + 2.4*b04 + 1)
      NGRDI = ((b03 - b04)/(b03 + b04 + eps))
      MNDWI = ((b03 - b11)/(b03 + b11 + eps))
      OSAVI = (b08 - b04) / (b08 + b04 + 0.35)
      WDRVI = (0.1 * b08 - b04)/(0.1 * b08 + b04 + eps)
      TGI =  b03 - 0.39 * b04 - 0.61 * b02
      GCVI = ( b08 / (b03 + eps) ) - 1
      RGVI = 1 - ( (b01 + b03) / (b04 + b05 + b07 + eps) )
      MI = (b8a - b11) / (b8a + b11 + eps)
      ARVI =  (b08 - (2 * b04) + b02) / (b08 + (2 * b04) + b02 + eps)
      SIPI = (b08 - b02) / (b08 - b04 + eps)
      RENDVI = (b06 - b05) / (b06 + b05 + eps) 
      MRESR = (b06 - b01) / (b05 - b01 + eps)
      RYI = b03 / (b02 + eps)
      NDYI = (b03 - b02) / (b03 + b02 + eps)
      DYI = b03 - b02
      ACI = b08 * (b04 + b03 + eps)
      CVI = (b08 / (b03 + eps)) * (b04 / (b03 + eps))
      AVI = (b08 * (1 - b04) * (b08 - b04))
      SI = ((1 - b02) * (1 - b03) * (1 - b04))
      BSI = ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02) + eps)
      SAVI = ((b08 - b04)/(b08 + b04 + 0.33)) * (1 + 0.33)
      FIDET = b12 / (b8a * b09  + eps)
      MTCI = (b06 - b05)/(b05 - b04 + eps)
      NPCRI = (b04 - b02) / (b04 + b02 + eps)
      S2REP = 705 + 35 * ((((b07 + b04)/2) - b05)/(b06 - b05 + eps)) 
      CCCI = ((b08 - b05) / (b08 + b05 + eps)) / ((b08 - b04) / (b08 + b04 + eps)) 
      MCARI = ((b05 - b04) - 2 * (b05 - b03)) * (b05 / (b04 + eps))  
      TCARI = 3 * ((b05 - b04) - 0.2 * (b05 - b03) * (b05 / (b04 + eps))) 
      PVI = (b08 - 0.3 * b04 - 0.5) / ((1 + 0.3 * 2) ** 0.5) 
      EVI = 2.5*(b08 - b04) / (b08 + 6*b04 - 7.5*b02 + 1 + eps)
      NDVI = ((b08 - b04)/(b08 + b04 + eps))
      BAI = 1/((0.1 - b04) ** 2 + (0.06 - b08) ** 2)
      MTVI2 = list(1.5*(1.2 * (i - j) - 2.5 * (k - j))* ((2 * i + 1)**2 - (6 * i - 5 * k ** (0.5)) - 0.5)**(0.5) for i, j, k in zip(b08, b03, b04))
      NDSI = (b03 - b11) / (b03 + b11 + eps)
      MRENDVI = (b06 - b05) / (b06 + b05 - 2 * b01 + eps) 
      NDVIre = (b08 - b05)/(b08 + b05 + eps)
      CIre = ((b08 / (b05 + eps)) - 1)
      NDMI = (b08 - b11)/(b08 + b11 + eps) 
      TNDVI = ((b08 - b04) / (b08 + b04 + eps) + 0.5) ** 0.5
      VDVI = (2 * b03 - b04 - b02) / (2 * b03 + b04  + b02 + eps)
      NBR = (b08 - b11) / (b08+ b11 + eps)
      TVI = (120 * (b06 - b03) - 200 * (b04 - b03)) / 2
      EXG = 2 * b03 - b04 - b02
      PSRI = (b04 - b02) / (b06 + eps)
      RDVI = (b08 - b04) / (b08 + b04 + eps)**0.5

      RATIO1 = b01 / (b03 + eps)
      RATIO2 = b01 / (b05 + eps)
      RATIO3 = b11 / (b12 + eps)
      RATIO4 = b05 / (b04 + eps)
      RATIO5 = b07 / (b05 + eps)
      RATIO6 = b07 / (b06 + eps)
      RATIO7 = b08 / (b04 + eps)

      return [MI, ARVI, SIPI, RENDVI, MRESR, RYI, NDYI, DYI, ACI, CVI, AVI, SI, BSI, SAVI, FIDET, MTCI, NPCRI, S2REP, CCCI,          MCARI, TCARI, PVI, EVI, NDVI, BAI, MTVI2, NDSI, MRENDVI, NDVIre, CIre, NDMI, TNDVI, VDVI, NBR, TVI, EXG, PSRI, RDVI,           NDRE, MMSR, NDWI, GNDVI, EVI2, NGRDI, MNDWI, OSAVI, WDRVI, TGI, GCVI, RGVI,
              RATIO1, RATIO2, RATIO3, RATIO4, RATIO5, RATIO6, RATIO7]


    ############################################################################################
    ######################################## Inference #########################################
    ############################################################################################


    selected_bands = ['B01', 'B02', 'B03', 'B04','B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12']
    spectral_indices = ['MI', 'ARVI', 'SIPI', 'RENDVI', 'MRESR', 'RYI', 'NDYI', 'DYI', 'ACI', 'CVI', 'AVI', 'SI', 'BSI',                    'SAVI', 'FIDET', 'MTCI', 'NPCRI', 'S2REP', 'CCCI', 'MCARI', 'TCARI', 'PVI', 'EVI', 'NDVI', 'BAI', 'MTVI2', 'NDSI', 'MRENDVI',                    'NDVIre', 'CIre', 'NDMI', 'TNDVI', 'VDVI', 'NBR', 'TVI', 'EXG', 'PSRI', 'RDVI', 
                        'NDRE', 'MMSR', 'NDWI', 'GNDVI', 'EVI2', 'NGRDI', 'MNDWI', 'OSAVI', 'WDRVI', 'TGI', 'GCVI', 'RGVI',
                        'RATIO1', 'RATIO2', 'RATIO3', 'RATIO4', 'RATIO5', 'RATIO6', 'RATIO7']
    n_selected_bands = len(selected_bands)

    test_competition_data = build_collection_df(test_label_collection)
    test_data = paralellize(tile_feature_extractor, zip(test_competition_data.unique_folder_id.values, test_competition_data.field_paths.values), size=len(test_competition_data), with_tqdm=True)
    test_data = tiles_to_df(test_data)
    test_data['field_id'] = test_data['field_id'].astype(int).astype(str)

    comb = test_data.copy()
    comb[selected_bands] = comb[selected_bands]**0.5

    vegs = vegetation_index()

    vi_df = pd.DataFrame()
    for n, m in zip(spectral_indices, vegs):
      vi_df[n] = m

    vi_df = vi_df.replace([np.inf, -np.inf], np.nan)

    test_data_full = pd.concat([comb[['field_id']].reset_index(drop=True), vi_df.reset_index(drop=True)], axis = 1)


    crop_names = [
            'Wheat', 'Mustard','Lentil','No Crop','Green pea', 'Sugarcane','Garlic','Maize','Gram','Coriander','Potato','Bersem','Rice'
        ]
    main_cols = test_data_full.columns.difference(['field_id', 'crop_id', 'fold', 'target'], sort=False).tolist()


    n_splits = 10
    model_list = [f'{model_path}pixelwise-lgbm-{n}.pkl' for n in range(n_splits)]

    oofs = []
    for model_name in tqdm(model_list):
        with open(model_name, 'rb') as f:
            model = pickle.load(f)

        oofs.append(
            model.predict_proba(test_data_full[main_cols])
        )

    subs = test_data_full[['field_id']].copy()
    subs['field_id'] = subs['field_id'].astype(int).astype(str)
    subs[crop_names] = np.mean(oofs, axis=0)
    subs = subs.groupby('field_id').mean().reset_index()
    subs.to_csv(f'{output_path}/pixelwise-lgbm.csv', index=False)
