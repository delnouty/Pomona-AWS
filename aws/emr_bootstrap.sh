#!/bin/bash
# Bootstrap EMR : installe sur TOUS les nœuds les librairies Python requises par les UDF.
# TensorFlow n'est pas installé ici : il vient de l'application EMR "TensorFlow"
# (emr-6.3.0 -> TF 2.4.1). Même liste que le mode opératoire P8.
# pandas est figé en 1.2.5 car le numpy livré avec EMR 6.3.0 est trop ancien pour pandas >= 1.3.
set -euxo pipefail

sudo python3 -m pip install -U setuptools
sudo python3 -m pip install -U pip
sudo python3 -m pip install wheel
sudo python3 -m pip install pillow
sudo python3 -m pip install pandas==1.2.5
sudo python3 -m pip install pyarrow
# boto3, s3fs, fsspec : lecture du parquet S3 avec pandas depuis le notebook
sudo python3 -m pip install boto3
sudo python3 -m pip install s3fs
sudo python3 -m pip install fsspec
