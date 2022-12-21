# AgriFieldNet Model for Crop Detection from Satellite Imagery

First place solution of the [Zindi AgriFieldNet India Challenge](https://zindi.africa/competitions/agrifieldnet-india-challenge).

Small farms produce about 35% of the world’s food, and are mostly found in low-
and middle-income countries. Reliable information about these farms is limited,
making support and policy-making difficult. Earth Observation data from
satellites such as Sentinel-2, in combination with machine learning, can help
improve agricultural monitoring, crop mapping, and disaster risk management for
these small farms. The Main goal of this challenge is to classify crop types in
agricultural fields across Northern India using multispectral observations from
Sentinel-2 satellite. These fields are located in various districts in states
of Uttar Pradesh, Rajasthan, Odisha and Bihar.

![model_ecaas_agrifieldnet_gold_v1](https://radiantmlhub.blob.core.windows.net/frontend-dataset-images/ref_agrifieldnet_competition_v1.png)

MLHub model id: `model_ecaas_agrifieldnet_gold_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/model_ecaas_agrifieldnet_gold_v1).

## Training Data

- [Training Data Source](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_source)
- [Training Data Labels](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_labels_train)

## Related MLHub Dataset

[AgriFieldNet Competition Dataset](https://mlhub.earth/data/ref_agrifieldnet_competition_v1)

## Citation

Muhamed T, Emelike C, Ogundare T, "AgriFieldNet Model for Crop Detection from
Satellite Imagery", Version 1.0, Radiant MLHub. [Date Accessed] Radiant MLHub
<https://doi.org/10.34911/rdnt.k2ft4a>

## License

[CC-BY-4.0](../LICENSE)

## Creators

- [Muhamed Tuo](https://www.linkedin.com/in/muhamed-tuo-b1b3a0162/)
- [Caleb Emelike](https://www.linkedin.com/in/caleb-emelike-6a040219a/)
- [Taiwo Ogundare](https://www.linkedin.com/in/taiwo-ogundare/)

## Contact

Muhamed Tuo [tuomuhamed@gmail.com](mailto:tuomuhamed@gmail.com)

## Applicable Spatial Extent

The applicable spatial extent, for new inferencing.

```geojson
{
    "type": "FeatureCollection",
    "features": [
        {
            "properties": {
                "id": "ref_agrifieldnet_competition_v1"
            },
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "bbox": [
                    76.2448,
                    18.9414,
                    88.046,
                    28.327
                ],
                "coordinates": [
                    [
                        [
                            [
                                88.046,
                                18.9414
                            ],
                            [
                                88.046,
                                28.327
                            ],
                            [
                                76.2448,
                                28.327
                            ],
                            [
                                76.2448,
                                18.9414
                            ],
                            [
                                88.046,
                                18.9414
                            ]
                        ]
                    ]
                ]
            }
        }
    ]
}
```

## Applicable Temporal Extent

The recommended start/end date of imagery for new inferencing.

| Start | End |
|-------|-----|
| 2022-01-01 | present |

## Learning Approach

- Supervised

## Prediction Type

- Classification

## Models Architecture

- Gradient Boosting (Catboost, Lightgbm, Xgboost)
- Unet

## Training Operating System

- Linux

## Training Processor Type

Both CPU and GPU.

Models trained on CPU:

- single-catboost
- single-xgboost
- crossval-catboost (40 cores TPU)
- pixelwise-lightgbm (40 cores TPU)

Models trained on GPU:

- R-model-catboost
- pixelwise-catboost
- fieldwise-catboost
- pixelwise-unet

## Model Inferencing

Review the [GitHub repository README](../README.md) to get started running
this model for new inferencing.

### Training

For the features engineering, we used bands
`("B01","B02","B03","B04","B05","B06","B07","B08","B09","B11", "B12")` and also
calculated a few derived bands using well known formulae.

The following are the derived indices:  

- NDVI (Normalized Green Red Difference Index) : `(B08 - B04)/ (B08 + B04)`
- GLI (Green Leaf Index) : `(2 * B03 - B04 - B02)/(2 * B03 + B04 + B02)`
- CVI : (Chlorophyll Vegetation Index) : `(B08 / B03) * (B04 / B03)`
- SIPI : `(B08 - B02) / (B08 - B04)`
- S2REP : `705 + 35 * (((( B07 + B04 ) / 2) - B05 ) / (B06 - B05))`
- CCCI : `((B08 - B05) / (B08 + B05)) / ((B08 - B04) / (B08 + B04))`
- HUE (Overall Hue Index) : `atan( 2 * ( B02 - B03 - B04 ) / 30.5 * ( B03 - B04 ))`
- RENDVI : `(B06 - B05) / (B06 + B05)`
- RECI `(Chlorophyll Index) : ( B08 / B04 ) - 1`
- EVI (Enhanced Vegetation Index) : `(2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0))`
- EVI2 (Enhanced Vegetation Index 2) : `(2.4 * (B08 - B04) / (B08 + B04 + 1.0))`
- NDWI : `(B04 - B02) / (B04 + B02)`
- NPCRI : `(B03 - B08) / (B03 + B08)`

Then we took median and max of the above features. We also calculated the total
area percentage of a given field using library
[FIELDimageR](https://github.com/OpenDroneMap/FIELDimageR), along with other
features like the field tile count, field overlap count, field tile size, field
tile height, field tile width.

### Model

It consist of 1 Unet + 8 Gradient Boosting Trees.

### Structure of Output Data

Each of the models will generate an output file (in csv). If a model is named
`single-model-agrifield.ext`, its corresponding output file will be
`single-model-agrifield.csv`. The final output file (`submission.csv`) is a
weighted geometric mean of all the intermediate output files.

## Winning Solution from AgrifieldNet India Challenge

The original solution code is archived in the file:
[first-place-agrifieldnet-solution.zip](first-place-agrifieldnet-solution.zip).
Please note: this repository uses [Git Large File Support
(LFS)](https://git-lfs.github.com/) to include this .zip file. Either install
git lfs support for your git client, use the official Mac or Windows GitHub
client to clone this repository.
