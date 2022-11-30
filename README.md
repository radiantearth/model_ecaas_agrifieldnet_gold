# AgriFieldNet Model for Crop Types Detection

First place solution for AgriFieldNet India Challenge Crop Types Detection from Satellite Imagery competition

![{{model_ecaas_agrifieldnet_gold_v1}}](https://radiantmlhub.blob.core.windows.net/frontend-dataset-images/ref_agrifieldnet_competition_v1.png)

MLHub model id: `model_ecaas_agrifieldnet_gold_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/{{model_id}}).

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

First clone this Git repository.

```bash
git clone https://github.com/radiantearth/model_ecaas_agrifieldnet_gold.git
cd model_ecaas_agrifieldnet_gold/
```

After cloning the model repository, you can use the Docker Compose runtime
files as described below.

## Pull or Build the Docker Image

Pull pre-built image from Docker Hub (recommended):

```bash
docker pull docker.io/radiantearth/model_ecaas_agrifieldnet_gold:1
```

Or build image from source:

```bash
docker build -t radiantearth/model_ecaas_agrifieldnet_gold:1 -f Dockerfile .
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

3. Run the appropriate Docker Compose command for your system

    ```bash
    docker compose up model_ecaas_agrifieldnet_gold_v1
    ```

4. Wait for the `docker compose` to finish running, then inspect the
`OUTPUT_DATA` folder for results.

## Understanding Output Data

Please review the model output format and other technical details in the [model
documentation](/docs/index.md).
