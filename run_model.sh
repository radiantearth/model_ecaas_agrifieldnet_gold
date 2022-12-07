#!/bin/bash -e

# Generate data
Rscript model_ecaas_agrifieldnet_gold/r-generate-data.R

# Inference
python model_ecaas_agrifieldnet_gold/r-catboost-inference.py
python model_ecaas_agrifieldnet_gold/single-catboost-inference.py
python model_ecaas_agrifieldnet_gold/single-xgboost-inference.py
python model_ecaas_agrifieldnet_gold/crossval-catboost-inference.py
python model_ecaas_agrifieldnet_gold/pixelwise-catboost-inference.py
python model_ecaas_agrifieldnet_gold/pixelwise-lightgbm-inference.py
python model_ecaas_agrifieldnet_gold/fieldwise-catboost-inference.py
python model_ecaas_agrifieldnet_gold/pixelwise-unet-inference.py

# Blend and create final submission file
python model_ecaas_agrifieldnet_gold/blending.py
