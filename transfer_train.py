from argparse import Namespace
from logging import Logger
import os
import pickle
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split

from crystalnet.train import transfer_train, transfer_evaluate
from crystalnet.models import build_model
from crystalnet.data.scaler import StandardScaler
from crystalnet.nn_utils import param_count
from crystalnet.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, makedirs, save_checkpoint
from crystalnet.data import CrystalDataset
from crystalnet.data.utils import get_task_names, get_data, get_class_sizes
from crystalnet.parsing import parse_train_args
from crystalnet.utils import create_logger
from crystalnet.features import AtomCustomJSONInitializer, GaussianDistance, load_radius_dict


def run_training(train_data: CrystalDataset, valid_data: CrystalDataset, args: Namespace, logger: Logger = None):

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Target adjust
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(train_data)
        info('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            info(f'{args.task_names[i]} '
                 f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.dataset_type == 'regression':
        info('Fitting scaler')
        train_targets = train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Get best validation loss
    best_validation_scores = None

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args)
        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # transfer learning, only transfer ffn
        for name, param in model.named_parameters():
            if name.find('ffn') != -1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler=None, args=args)

        # Optimizers
        optimizer = build_optimizer(model.ffn, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0

        for epoch in range(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = transfer_train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            # (num_tasks,)
            valid_scores = transfer_evaluate(
                model=model,
                data=valid_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(valid_scores)
            debug(f'Valid {args.metric} = {avg_val_score:.6f}')
            writer.add_scalar(f'valid_{args.metric}', avg_val_score, n_iter)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, valid_scores):
                    debug(f'Valid {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'valid_{task_name}_{args.metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or not args.minimize_score and avg_val_score > best_score:
                best_validation_scores = valid_scores
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler=None, args=args)

        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best valid {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        if args.show_individual_scores:
            # Individual validation scores
            for task_name, val_score in zip(args.task_names, best_validation_scores):
                debug(f'Valid {task_name} {args.metric} = {val_score:.6f}')
                writer.add_scalar(f'valid_{task_name}_{args.metric}', val_score, n_iter)


def transfer(args: Namespace, logger: Logger = None):
    info = logger.info if logger is not None else print

    # Load feature object
    ari = AtomCustomJSONInitializer(f'{args.data_path}/atom_init.json')
    dmin, dmax, step, var = args.rbf_parameters
    gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step, var=var)
    radius_dic = load_radius_dict(f'{args.data_path}/hubbard_u.yaml')

    # Load and cache data
    info('Loading data')
    if os.path.exists(f'{args.train_path}/graph_cache.pickle'):
        with open(f'{args.train_path}/graph_cache.pickle', 'rb') as f:
            all_graph = pickle.load(f)
    else:
        assert "There is no poscar graph cache, please use preprocess.py to generate poscar graph cache!"

    if os.path.exists(f'{args.train_path}/transfer.pickle'):
        with open(f'{args.train_path}/transfer.pickle', 'rb') as f:
            all_data = pickle.load(f)
    else:
        all_data = get_data(path=f'{args.train_path}/experiment_band_gap.csv',
                            graph=all_graph, ari=ari, gdf=gdf, radius_dic=radius_dic, args=args, logger=logger)
        with open(f'{args.train_path}/transfer.pickle', 'wb') as fw:
            pickle.dump(all_data, fw)

    # Split data
    train_data, valid_data = train_test_split(all_data, test_size=0.1, random_state=args.run_fold)
    train_data, valid_data = CrystalDataset(train_data), CrystalDataset(valid_data)
    args.task_names = get_task_names(path=f'{args.train_path}/experiment_band_gap.csv', use_compound_names=True)
    # fake the num_tasks for loading the pretrain model
    args.num_tasks = train_data.num_tasks()

    info(f'Number of tasks = {args.num_tasks}')
    info(f'Total size = {len(train_data)+len(valid_data):,} | '
         f'train size = {len(train_data):,}({len(train_data)/(len(train_data)+len(valid_data)):.1f}) | '
         f'valid size = {len(valid_data):,}({len(valid_data)/(len(train_data)+len(valid_data)):.1f})')

    # Required for NormLR
    args.train_data_size = len(train_data)

    # Training
    run_training(train_data, valid_data, args, logger)


if __name__ == '__main__':
    args = parse_train_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    transfer(args, logger)
