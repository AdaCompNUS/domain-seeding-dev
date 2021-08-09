```
git submodule init
git submodule update --init --recursive iGibson
conda create -n fetch_push_env python=3.8
conda activate fetch_push_env
cd src/iGibson
pip install -e .
python -m igibson.utils.assets_utils --download_assets
cd ../
python fetch_gym.py
```
