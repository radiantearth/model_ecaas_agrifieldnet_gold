# AgriFieldNet Model for Crop Types Detection

First place solution for AgriFieldNet India Challenge Crop Types Detection from Satellite Imagery competition

![{{model_ecaas_agrifieldnet_gold_v1}}](https://radiantmlhub.blob.core.windows.net/frontend-dataset-images/ref_agrifieldnet_competition_v1.png)

MLHub model id: `model_ecaas_agrifieldnet_gold_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/model_ecaas_agrifieldnet_gold_v1).

## ML Model Documentation

Please review the model architecture, license, applicable spatial and temporal extents
and other details in the [model documentation](/docs/index.md).

## System Requirements

* Git client
* [Docker](https://www.docker.com/) with
    [Compose](https://docs.docker.com/compose/) v1.28 or newer.

## Hardware Requirements

||Training|Inferencing|
|---|-----------|--------|
|RAM|25 GB RAM | 16 GB RAM|
|NVIDIA GPU| Required | Optional (but very slow for Unet model)|

## Get Started With Inferencing

Start by cloning this repository to get the model scripts and saved model
checkpoint files:

```bash
git clone https://github.com/radiantearth/model_ecaas_agrifieldnet_gold.git
cd model_ecaas_agrifieldnet_gold/
```

To get started, the R and Python dependencies must to be installed locally in
your environment. Alternatively you can look at the original AgriFieldNet
Challenge solution scripts (.zip) which are linked in the [model
documentation](./docs/index.md).

### R and packages

[R 4.2.2](https://www.r-project.org/) is required for the data preprocessing and
feature engineering step of the model.

```bash
R -e "install.packages(c('devtools', 'plyr', 'tidyverse', 'raster', 'celestial', 'caret', 'fastICA', 'SOAR', 'RStoolbox', 'jsonlite', 'data.table', 'spdep'))"
R -e "devtools::install_github('OpenDroneMap/FIELDimageR')"
```

### Python and packages

[Python 3.8](https://www.python.org/) is required for the model training and inferencing steps.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Model to Generate New Inferences

1. Prepare your input and output data folders. The `data/` folder in this repository
    contains some placeholder files to guide you.

    * The `data/` folder must contain:
        * `input/ref_agrifieldnet_competition_v1`: for inferencing:
            * `ref_agrifieldnet_competition_v1_source`: folder containing for the satellite imagery bands
                * `ref_agrifieldnet_competition_v1_source_{folder_id}`: e.g. `ref_agrifieldnet_competition_v1_source_0a1d5`
            * `ref_agrifieldnet_competition_v1_labels_test`: containing the field ids
                * `ref_agrifieldnet_competition_v1_labels_test_{folder_id}`: e.g. `ref_agrifieldnet_competition_v1_labels_test_0a1d5`
    * The `output/` folder is where the model will write inferencing results.

2. Set `INPUT_DATA` and `OUTPUT_DATA` environment variables corresponding with
    your input and output folders. These commands will vary depending on operating
    system and command-line shell:

    ```bash
    # change paths to your actual input and output folders
    export INPUT_DATA="/home/my_user/model_ecaas_agrifieldnet_gold/data/input/"
    export OUTPUT_DATA="/home/my_user/model_ecaas_agrifieldnet_gold/data/output/"
    ```

3. Run the `run_models.sh` bash shell script.

    ```bash
    ./run_models.sh
    ```

4. Wait for the script to finish running, then inspect the `OUTPUT_DATA` folder
for results. If you run into errors, or missing packages, alternatively you can look at the
original AgriFieldNet Challenge solution scripts (.zip) which are linked in the
[model documentation](./docs/index.md).

## Understanding Output Data

Please review the model output format and other technical details in the [model
documentation](/docs/index.md).
