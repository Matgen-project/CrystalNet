# CrystalNet
CrystalNet is a directed graph-based deep learning model for predicting material properties based on the CMPNN[1] framework.

## Requirement
```
ase                   3.21.1
cached-property       1.5.2
certifi               2020.12.5
chardet               4.0.0
cycler                0.10.0
decorator             4.4.2
future                0.18.2
googledrivedownloader 0.4
greenlet              1.0.0
h5py                  3.2.1
idna                  2.10
importlib-metadata    4.0.1
isodate               0.6.0
Jinja2                2.11.3
joblib                1.0.1
kiwisolver            1.3.1
llvmlite              0.36.0
MarkupSafe            1.1.1
matplotlib            3.4.1
monty                 2021.5.9
mpmath                1.2.1
networkx              2.5.1
numba                 0.53.1
numpy                 1.20.2
olefile               0.46
palettable            3.3.0
pandas                1.2.4
Pillow                8.2.0
pip                   20.1.1
plotly                4.14.3
plyfile               0.7.3
protobuf              3.17.0
pycairo               1.20.0
pymatgen              2020.12.18
pyparsing             2.4.7
python-dateutil       2.8.1
python-louvain        0.15
pytz                  2021.1
rdflib                5.0.0
reportlab             3.5.67
requests              2.25.1
retrying              1.3.3
ruamel.yaml           0.17.4
ruamel.yaml.clib      0.2.2
scikit-learn          0.24.1
scipy                 1.6.3
setuptools            52.0.0.post20210125
six                   1.15.0
spglib                1.16.1
SQLAlchemy            1.4.11
sympy                 1.8
tabulate              0.8.9
tensorboardX          2.2
threadpoolctl         2.1.0
torch                 1.5.0+cu101
torch-cluster         1.5.5
torch-geometric       1.5.0
torch-scatter         2.0.5
torch-sparse          0.6.6
torch-spline-conv     1.2.0
torchaudio            0.5.0
torchvision           0.6.0+cu101
tornado               6.1
tqdm                  4.60.0
typing-extensions     3.7.4.3
uncertainties         3.1.5
urllib3               1.26.4
wheel                 0.36.2
zipp                  3.4.1
```


## How to prepare dataset?
Specified the fowllowing files path in proprecess.py
- property.csv
- poscar files

And then run proprecess.py.
## How to run?
python -u train.py --gpu 0 --seed 333 --data_path ../data/preprocess --train_path ./data/preprocess/calculate --dataset_name band_gap --dataset_type regression --run_fold 0 --metric mae --save_dir ./ckpt/ensemble_band_gap --epochs 200 --init_lr 1e-4 --max_lr 3e-4 --final_lr 1e-4 --no_features_scaling --show_individual_scores --max_num_neighbors 64  > ./log/fold_0_band_gap.log 2>&1 &

## How to predict properties?
python -u predict.py --gpu 0 --seed 0 --data_path ./data/matgen/preprocess --test_path ./data/matgen/preprocess/calculate --dataset_name band_gap --checkpoint_dir ./result/20210520/ckpt/ensemble/ --no_features_scaling > ./log/predict_band_gap.log 2>&1 &


## How to transfer lerning?
python -u transfer_train.py --gpu 0 --data_path ./data/preprocess --train_path ./data/preprocess/experiment_1716 --dataset_name band_gap --dataset_type regression --metric mae --run_fold 0 --save_dir ./ckpt/transfer/fold_1 --checkpoint_dir ./result/0603/ensemble_band_gap/ --epochs 200 --init_lr 1e-4 --max_lr 3e-4 --final_lr 1e-4 --no_features_scaling --show_individual_scores > ./log/fold_0_transfer.log 2>&1 &


### Reference:
[1]. Ying Song, Shuangjia Zheng, Zhangming Niu, Zhang-Hua Fu, Yutong Lu, Yuedong Yang: Communicative Representation Learning on Attributed Molecular Graphs. IJCAI 2020: 2831-2838

