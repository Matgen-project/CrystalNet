import os
import pickle
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from pymatgen.io.vasp import Poscar
from tqdm import tqdm


def generate_graph_cache(poscar_path, save_path, save_name):
    all_data = {crystal_name: Poscar.from_file(os.path.join(poscar_path, crystal_name)).as_dict()
                for crystal_name in tqdm(os.listdir(poscar_path))}
    with open(os.path.join(save_path, f'{save_name}.pickle'), 'wb') as f:
        pickle.dump(all_data, f)


def split_and_save_data(file_path, seed):
    kfold = KFold(n_splits=9, shuffle=True, random_state=seed)

    if not os.path.exists(f'./calculate/seed_{seed}/'):
        os.makedirs(f'./calculate/seed_{seed}')

    data = pd.read_csv(file_path)
    train_val_data, test_data = train_test_split(data, test_size=0.1, random_state=seed)
    test_data.to_csv(f'./calculate/seed_{seed}/test.csv', index=None)
    for fold_num, (train_index, valid_index) in enumerate(kfold.split(train_val_data)):
        train_data, valid_data = train_val_data.iloc[train_index], train_val_data.iloc[valid_index]
        train_data.to_csv(f'./calculate/seed_{seed}/train_fold_{fold_num + 1}.csv', index=None)
        valid_data.to_csv(f'./calculate/seed_{seed}/valid_fold_{fold_num + 1}.csv', index=None)


def fine_tune_split_data(file_path, seed):
    kfold = KFold(n_splits=9, shuffle=True, random_state=seed)

    if not os.path.exists(f'./seed_{seed}/'):
        os.makedirs(f'./seed_{seed}')

    data = pd.read_csv(file_path)
    for fold_num, (train_index, valid_index) in enumerate(kfold.split(data)):
        train_data, valid_data = data.iloc[train_index], data.iloc[valid_index]
        train_data.to_csv(f'./seed_{seed}/finetune_train_fold_{fold_num + 1}.csv', index=None)
        valid_data.to_csv(f'./seed_{seed}/finetune_valid_fold_{fold_num + 1}.csv', index=None)


if __name__ == "__main__":
    #split_and_save_data('./calculate/property_rm_outliers.csv', seed=333)
    #split_and_save_data('./calculate/property.csv', seed=333)
     generate_graph_cache(poscar_path='./poscar_size_influence/seed_333/400_big', save_path='./poscar_size_influence/seed_333/',
                          save_name='graph_cache_big_size')








