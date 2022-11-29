#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob
import numpy as np
import pandas as pd
import scipy
from scipy.stats import gmean


if __name__ == '__main__':

  ############################################################
  ######################## Utilities #########################
  ############################################################
      
  def read_csv(fs):
      return [
          pd.read_csv(f'{sub_path}/{csv_file}') \
          .rename(columns={'Field ID': 'field_id'}) \
          .sort_values(by=['field_id']) \
          .reset_index(drop=True)
          
          for csv_file in fs
      ]

  def blend(dfs, f=gmean, w=None):
      df = dfs[0].copy()
      
      data = [
          df_[cols].values for df_ in dfs
      ]
      
      if w is None:
          df[cols] = f(data, axis=0)
      else:
          df[cols] = f(data, weights=w, axis=0)
      
      return df

  ##########################################################
  ###################### Blending ##########################
  ##########################################################

  sub_path = os.environ['OUTPUT_DATA']


  cols = [
      'Wheat', 'Mustard', 'Lentil', 'No Crop', 'Green pea', 'Sugarcane', 
      'Garlic', 'Maize', 'Gram', 'Coriander', 'Potato', 'Bersem', 'Rice'
  ]


  dfs_1 = read_csv(
      [
          'single-catboost.csv',
          'single-xgboost.csv',
      ]
  )

  dfs_2 = read_csv(
      [
          'crossval-catboost.csv',
          'subm.csv',
      ]
  )

  dfs_3 = read_csv(
      [
          'fieldwise-catboost.csv',
          'pixelwise-catboost.csv',
          'pixelwise-lgbm.csv',
          'pixelwise-unet.csv',
      ]
  )

  weights_1 = np.array([
      np.ones_like(dfs_1[0][cols].values) * 0.7,
      np.ones_like(dfs_1[0][cols].values) * 0.3,
  ])
  weights_2 = np.array([
      np.ones_like(dfs_1[0][cols].values) * 0.55,
      np.ones_like(dfs_1[0][cols].values) * 0.45,
  ])
  weights_3 = np.array([
      np.ones_like(dfs_1[0][cols].values) * 0.45,
      np.ones_like(dfs_1[0][cols].values) * 0.15,
      np.ones_like(dfs_1[0][cols].values) * 0.35,
      np.ones_like(dfs_1[0][cols].values) * 0.05,
  ])


  blend_1 = blend(dfs_1, f=np.average, w=weights_1)
  blend_2 = blend(dfs_2, f=np.average, w=weights_2)
  blend_3 = blend(dfs_3, f=np.average, w=weights_3)

  dfs = [
      blend_1,
      blend_2,
      blend_3,
  ]

  weights = np.array([
      np.ones_like(dfs[0][cols].values) * 0.85,
      np.ones_like(dfs[0][cols].values) * 0.1,
      np.ones_like(dfs[0][cols].values) * 0.05,
  ])


  df_blend = blend(dfs, w=weights)
  
  df_blend.to_csv(f'{sub_path}/submission.csv', index=False)


