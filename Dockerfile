# R builds
FROM r-base:4.2.2

run apt-get update && \
  apt-get install -y libcurl4-openssl-dev libssl-dev libssh2-1-dev libxml2-dev zlib1g-dev && \
  R -e "install.packages(c('devtools', 'plyr', 'tidyverse', 'raster', 'celestial', 'caret', 'fastICA', 'SOAR', 'RStoolbox', 'jsonlite', 'data.table', 'spdep'))"


RUN R -e "devtools::install_github('OpenDroneMap/FIELDimageR')"

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y && \
    pip install --upgrade pip
    
COPY requirements.txt /tmp/requirements-docker.txt

ENV INPUT_DATA="./data/input"
ENV OUTPUT_DATA="./data/output"

RUN pip install -r /tmp/requirements-docker.txt && \
    rm /tmp/requirements-docker.txt

RUN mkdir -p /app
WORKDIR /app

COPY . .

ENTRYPOINT [ "bash", "run_model.sh" ]
