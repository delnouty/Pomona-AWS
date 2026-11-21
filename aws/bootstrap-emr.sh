#!/bin/bash
# EMR bootstrap action: runs on EVERY node at start-up, before Spark.
# Installs exactly the library versions tested locally (notebook section 3.3).
set -euxo pipefail
python3 --version
python3 -m pip --version || sudo dnf install -y python3-pip
sudo python3 -m pip install tensorflow-cpu==2.16.2 keras==3.5.0 numpy==1.26.4 \
                            pandas==2.2.3 pyarrow==17.0.0 pillow==10.4.0
python3 -c "import tensorflow as tf; print('bootstrap OK', tf.__version__)"
