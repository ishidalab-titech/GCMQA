import argparse
import json
import os
import re
import sys
from pathlib import Path

import chainer
import chainer.functions as F
import chainer.iterators as I
import matplotlib
import numpy as np
from chainer import training
from chainer.training import extensions

from dataproc import Dataproc
from evaluator import GraphEvaluator
from model import build_model, Classifier


# Global error handler
def global_except_hook(exctype, value, traceback):
    import sys
    from traceback import print_exception
    print_exception(exctype, value, traceback)
    sys.stderr.flush()

    import mpi4py.MPI
    mpi4py.MPI.COMM_WORLD.Abort(1)


sys.excepthook = global_except_hook


def main():
    import chainermn
    chainer.global_config.autotune = True
    parser = argparse.ArgumentParser(description='ChainerMN example: Train MQAP using 3DCNN')
    parser.add_argument('--communicator', type=str,
                        default='hierarchical', help='Type of communicator')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume the training from snapshot')
    parser.add_argument('--weight', '-w', action='store_true',
                        help='Resume only weight')
    parser.add_argument('--config', '-c', type=int, default=0,
                        help='Number of config')
    parser.add_argument('--config_file', type=str, default='./data/lddt_config.json',
                        help='Config file path')

    args = parser.parse_args()
    if args.gpu:
        if args.communicator == 'naive':
            print("Error: 'naive' communicator does not support GPU.\n")
            exit(-1)
        comm = chainermn.create_communicator(args.communicator, allreduce_grad_dtype='float16')
        device = comm.intra_rank
    else:
        if args.communicator != 'naive':
            print('Warning: using naive communicator '
                  'because only naive supports CPU-only execution')
        comm = chainermn.create_communicator('naive')
        device = -1
    f = open(args.config_file, 'r')

    config = json.load(f)['Config'][args.config]
    args.out = os.path.join(args.out, str(args.config))
    if comm.rank == 0:
        print('==========================================')
        chainer.print_runtime_info()
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num epoch: {}'.format(config['epoch']))
        print('Batch size:  {}'.format(config['batch_size'] * comm.size))
        print('Optimizer:  {}'.format(config['optimizer']))
        print('Learning Rate:  {}'.format(config['learning_rate']))
        print('Out Directory:  {}'.format(args.out))
        print('Vertex feature:  {}'.format(config['vertex_feature']))
        if config['global_mode']:
            print('Using Global loss')
        if config['local_mode']:
            print('Using local loss')
            print('Local type : {}'.format(config['local_type']))
            print('Local label : {}'.format(config['local_label']))
        print('==========================================')
    d = Dataproc(size=comm.size, rank=comm.rank, config=config)
    if device >= 0:
        chainer.cuda.get_device(device).use()
    # sub_comm = comm.split(comm.rank // comm.intra_size, comm.rank)
    if config['local_type'] == 'Regression':
        local_loss_func = F.mean_squared_error
    else:
        local_loss_func = F.sigmoid_cross_entropy
    global_loss_func = F.mean_squared_error
    model = build_model(config=config, comm=comm)
    model = Classifier(predictor=model, local_loss_func=local_loss_func, global_loss_func=global_loss_func,
                       config=config)
    if device >= 0:
        model.to_gpu()
    train, test = d.get_dataset(key='train'), d.get_dataset(key='test')
    train_iter = I.SerialIterator(dataset=train, batch_size=config['batch_size'], repeat=True, shuffle=True)
    test_iter = I.SerialIterator(dataset=test, batch_size=config['batch_size'], repeat=False, shuffle=False)
    # train_iter = I.MultiprocessIterator(dataset=train, batch_size=args.batch, repeat=True, shuffle=True, n_processes=10)
    # test_iter = I.MultiprocessIterator(dataset=test, batch_size=args.batch, repeat=False, shuffle=True, n_processes=10)

    if config['optimizer'] == 'Adam':
        optimizer = chainer.optimizers.Adam(alpha=config['learning_rate'],
                                            weight_decay_rate=config['weight_decay_rate'], amsgrad=True)
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm, double_buffering=False)
    elif config['optimizer'] == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD(lr=config['learning_rate'])
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm, double_buffering=False)
    elif config['optimizer'] == 'SMORMS3':
        optimizer = chainer.optimizers.SMORMS3(lr=config['learning_rate'])
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm, double_buffering=False)
    elif config['optimizer'] == 'Eve':
        from my_optimizer.eve import Eve, create_multi_node_optimizer
        optimizer = Eve(alpha=config['learning_rate'])
        optimizer = create_multi_node_optimizer(optimizer, comm, double_buffering=False)
    elif config['optimizer'] == 'Adabound':
        from my_optimizer.adabound import Adam as Adabound
        optimizer = Adabound(alpha=config['learning_rate'], adabound=True, amsgrad=True,
                             weight_decay_rate=config['weight_decay_rate'])
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm, double_buffering=False)
    optimizer.setup(model)
    val_interval = 1, 'epoch'
    log_interval = 1, 'epoch'
    updater = training.StandardUpdater(train_iter, optimizer, device=device, converter=d.get_converter())
    trainer = training.Trainer(updater, (config['epoch'], 'epoch'), out=args.out)
    evaluator = GraphEvaluator(iterator=test_iter, target=model.predictor, device=device, converter=d.get_converter(),
                               comm=comm, local_loss_func=local_loss_func, global_loss_func=global_loss_func,
                               name='val', config=config)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)
    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.snapshot(), trigger=val_interval)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], 'epoch', file_name='loss.png'),
                       trigger=val_interval)
        report_list = ['epoch', 'main/loss', 'val/main/loss']
        if config['global_mode']:
            report_list.extend(['main/global_loss', 'val/main/global_loss', 'val/main/global_pearson'])
            trainer.extend(extensions.PlotReport(['main/global_loss', 'val/main/global_loss'], 'epoch',
                                                 file_name='global_loss.png'), trigger=val_interval)
        if config['local_mode']:
            report_list.extend(['main/local_loss', 'val/main/local_loss', 'val/main/local_mean_pearson'])
            if config['local_type'] == 'Classification':
                report_list.append('val/main/local_auc')
                trainer.extend(extensions.PlotReport(['val/main/local_auc'], 'epoch', file_name='local_auc.png'),
                               trigger=val_interval)
            else:
                report_list.append('val/main/local_pearson')
        report_list.append('elapsed_time')
        trainer.extend(extensions.PrintReport(report_list), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))
    if args.resume:
        snap_list = [p for p in os.listdir(args.out) if 'snapshot' in p]
        snap_num = np.array([int(re.findall("[+-]?[0-9]+[\.]?[0-9]*[eE]?[+-]?[0-9]*", p)[0]) for p in snap_list])
        path = snap_list[np.argmax(snap_num)]
        path = os.path.join(args.out, path)
        if args.weight:
            obj_path = 'updater/model:main/predictor/'
            chainer.serializers.load_npz(path, model.predictor, obj_path)
        else:
            chainer.serializers.load_npz(path, trainer)
    if comm.rank == 0:
        protein_name_dict = d.get_protein_name_dict()
        out_path = Path(args.out)
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
        np.savez(os.path.join(args.out, 'protein_name'), **protein_name_dict)
        f = open(os.path.join(args.out, 'lddt_config.json'), 'w')
        json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
        f.close()
        f = open(os.path.join(args.out, 'args.json'), 'w')
        json.dump(vars(args), f)
        f.close()
    if comm.rank == 0:
        print('train start!!!')
    trainer.run()


if __name__ == '__main__':
    import multiprocessing as mp

    matplotlib.use('Agg')
    mp.set_start_method('forkserver', force=True)
    main()
