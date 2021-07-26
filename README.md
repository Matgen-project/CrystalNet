# CrystalNet

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

