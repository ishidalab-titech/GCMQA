import chainer
import chainer.functions as F
import training.graph_convolution as GC
import chainer.links as L
import chainermn.links as MNL
import numpy as np
from chainer import Variable
from chainer import link
from chainer import Sequential
import copy
from functools import partial
from chainer import function


def _get_model(layers, comm, predict=False):
    model = Sequential()
    W = chainer.initializers.HeNormal(1 / np.sqrt(1 / 2), dtype=np.float32)
    bias = chainer.initializers.Zero(dtype=np.float32)

    for layer in layers:
        name = layer['name']
        parameter = copy.deepcopy(layer['parameter'])
        if name.split('.')[0] == 'GC':
            parameter.update({'predict': predict})
            if 'batch_norm' in parameter.keys() and parameter['batch_norm']:
                parameter.update({'comm': comm})
            if 'activation' in parameter.keys() and parameter['activation']:
                parameter['activation'] = eval(parameter['activation'])
            add_layer = eval(name)(**parameter)
        elif name.split('.')[0] == 'L':
            if 'Linear' in name.split('.')[1]:
                parameter.update({'initialW': W, 'initial_bias': bias})
            add_layer = eval(name)(**parameter)
        elif name.split('.')[0] == 'F':
            if len(parameter) == 0:
                add_layer = partial(eval(name))
            else:
                add_layer = partial(eval(name), **parameter)
        elif name.split('.')[0] == 'MNL':
            if predict:
                add_layer = L.BatchNormalization(size=parameter['size'])
            else:
                add_layer = MNL.MultiNodeBatchNormalization(size=parameter['size'], comm=comm)
        elif name == 'Flat':
            add_layer = partial(lambda *x: x[0])
        model.append(add_layer)
    return model


def build_model(config, comm, predict=False):
    stem_model = _get_model(layers=config['model']['stem_model'], comm=comm, predict=predict)
    local_model = LocalModel(local_model=_get_model(layers=config['model']['local_model'], comm=comm, predict=predict),
                             out_size=len(config['local_label']))
    global_message = _get_model(layers=config['model']['global_message'], comm=comm, predict=predict)
    global_readout = _get_model(layers=config['model']['global_readout'], comm=comm, predict=predict)
    global_model = GlobalModel(out_size=len(config['global_label']), global_message=global_message,
                               global_readout=global_readout)
    model = Model(local_model=local_model, global_model=global_model, stem_model=stem_model)
    return model


class GlobalModel(chainer.Chain):
    def __init__(self, out_size, global_message, global_readout):
        super(GlobalModel, self).__init__()
        W = chainer.initializers.HeNormal(1 / np.sqrt(1 / 2), dtype=np.float32)
        bias = chainer.initializers.Zero(dtype=np.float32)
        with self.init_scope():
            self.global_message = global_message
            self.global_readout = global_readout
            self.out = L.Linear(out_size=out_size, in_size=None, initial_bias=bias, initialW=W)

    def __call__(self, vertex, edge, adj, num_array, batch_indices):
        global_vertex = self.global_message(vertex, edge, adj, num_array)
        global_feature = F.vstack([F.mean(i, axis=0) for i in F.split_axis(global_vertex, batch_indices[:-1], axis=0)])
        global_score = F.sigmoid(self.out(self.global_readout(global_feature)))
        return global_score


class LocalModel(chainer.Chain):
    def __init__(self, out_size, local_model):
        super(LocalModel, self).__init__()
        W = chainer.initializers.HeNormal(1 / np.sqrt(1 / 2), dtype=np.float32)
        bias = chainer.initializers.Zero(dtype=np.float32)
        with self.init_scope():
            self.local_model = local_model
            self.out = L.Linear(out_size=out_size, in_size=None, initial_bias=bias, initialW=W)

    def __call__(self, vertex, edge, adj, num_array):
        local_feature = self.local_model(vertex, edge, adj, num_array)
        local_out = self.out(local_feature)
        return local_out


class Model(chainer.Chain):
    def __init__(self, local_model, global_model, stem_model):
        super(Model, self).__init__()
        with self.init_scope():
            self.stem_model = stem_model
            self.local_model = local_model
            self.global_model = global_model

    def predict(self, vertex, edge, adj, num_array, batch_indices):
        with function.no_backprop_mode(), chainer.using_config('train', False):
            local_score, global_score = self.__call__(vertex, edge, adj, num_array, batch_indices)
            local_score = F.sigmoid(local_score)
            return local_score, global_score

    def __call__(self, vertex, edge, adj, num_array, batch_indices):
        vertex, edge, adj, num_array = self.stem_model(vertex, edge, adj, num_array)
        local_score = self.local_model(vertex=vertex, edge=edge, adj=adj, num_array=num_array)
        global_score = self.global_model(vertex=vertex, edge=edge, adj=adj, num_array=num_array,
                                         batch_indices=batch_indices)
        return local_score, global_score


class Classifier(link.Chain):
    def __init__(self, predictor, local_loss_func, global_loss_func, config):
        super(Classifier, self).__init__()
        self.local_loss_func = local_loss_func
        self.global_loss_func = global_loss_func
        self.config = config
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, vertex, edge, adj, num_array, batch_indices, local_label, global_label):
        local_score, global_score = self.predictor(vertex=vertex, edge=edge, adj=adj, num_array=num_array,
                                                   batch_indices=batch_indices)

        # Compute local loss
        loss = 0
        if self.config['local_mode']:
            if self.config['local_type'] == 'Classification':
                local_loss = self.local_loss_func(local_score, local_label)
            else:
                local_loss = self.local_loss_func(F.sigmoid(local_score), local_label)
            local_loss.name = 'Local Loss'
            chainer.reporter.report({'local_loss': local_loss}, self)
            loss += local_loss

        if self.config['global_mode']:
            # Compute global loss
            global_loss = self.global_loss_func(Variable(global_label), global_score)
            global_loss.name = 'Global Loss'
            chainer.reporter.report({'global_loss': global_loss}, self)

            # Sum local loss and global loss
            loss += global_loss
        loss.name = 'Loss'
        chainer.reporter.report({'loss': loss}, self)
        return loss


if __name__ == '__main__':
    import json

    config = json.load(open('./data/config.json', 'r'))['Config'][0]
    model = _get_model(layers=config['stem_model'], comm=None)
    print(model)