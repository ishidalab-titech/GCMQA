import chainer
import chainer.functions as F
import training.graph_convolution as GC
import chainer.links as L
import numpy as np
from chainer import Variable
from chainer import link
from chainer import Sequential
import copy
from functools import partial
from chainer import function


# from chainer_chemistry.links import Set2Set


def _get_model(layers, comm, predict=False):
    model = Sequential()
    W = chainer.initializers.HeNormal(1 / np.sqrt(1 / 2), dtype=np.float32)
    bias = chainer.initializers.Zero(dtype=np.float32)
    for layer in layers:
        name = layer['name']
        parameter = copy.deepcopy(layer['parameter'])
        if name.split('.')[0] == 'GC':
            if name.split('.')[1] == 'NodeAverage':
                parameter.update({'initialW': W, 'initial_bias': bias})
            add_layer = eval(name)(**parameter)
        elif name.split('.')[0] == 'L':
            if 'Linear' in name.split('.')[1]:
                parameter.update({'initialW': W, 'initial_bias': bias})
            add_layer = eval(name)(**parameter)
        elif name.split('.')[0] == 'F':
            add_layer = partial(eval(name), **parameter)
        elif name == 'Flat':
            add_layer = partial(lambda *x: x[0])
        elif name == 'FuncReadout':
            parameter['func'] = eval(parameter['func'])
            add_layer = eval(name)(**parameter)
        else:
            add_layer = eval(name)(**parameter)
        model.append(add_layer)
    return model


def build_model(config, comm, predict=False):
    stem_model = _get_model(layers=config['model']['stem_model'], comm=comm,
                            predict=predict)
    local_model = LocalModel(
        local_model=_get_model(layers=config['model']['local_model'], comm=comm,
                               predict=predict),
        out_size=len(config['local_label']))
    global_message = _get_model(layers=config['model']['global_message'],
                                comm=comm, predict=predict)
    global_readout = _get_model(layers=config['model']['global_readout'],
                                comm=comm, predict=predict)
    global_model = GlobalModel(out_size=1, global_message=global_message,
                               global_readout=global_readout)
    model = Model(local_model=local_model, global_model=global_model,
                  stem_model=stem_model)
    return model


class GlobalModel(chainer.Chain):
    def __init__(self, out_size, global_message, global_readout):
        super(GlobalModel, self).__init__()
        W = chainer.initializers.HeNormal(1 / np.sqrt(1 / 2), dtype=np.float32)
        bias = chainer.initializers.Zero(dtype=np.float32)
        with self.init_scope():
            self.global_message = global_message
            self.global_readout = global_readout
            self.out = L.Linear(out_size=out_size, in_size=None,
                                initial_bias=bias, initialW=W)

    def __call__(self, vertex, edge, adj, length):
        vertex = self.global_message(vertex, adj, edge)
        global_feature = self.global_readout(vertex, length)
        global_score = F.sigmoid(self.out(global_feature))
        return global_score


class FuncReadout(chainer.Chain):
    def __init__(self, func=F.mean):
        super(FuncReadout, self).__init__()
        self.func = func

    def __call__(self, vertex, length):
        global_feature = [v[:l] for v, l in zip(vertex, length)]
        global_feature = F.vstack(
            [self.func(i, axis=0) for i in global_feature])
        return global_feature


class Set2SetReadout(chainer.Chain):
    def __init__(self, in_channels, n_layers=1, processing_steps=3):
        super(Set2SetReadout, self).__init__()
        self.in_channels = in_channels
        self.n_layers = n_layers
        self.processing_steps = processing_steps
        with self.init_scope():
            self.set2set = Set2Set(in_channels=in_channels, n_layers=n_layers)

    def __call__(self, vertex, length):
        self.set2set.reset_state()
        for i in range(self.processing_steps):
            g = self.set2set(vertex)
        return g


class LocalModel(chainer.Chain):
    def __init__(self, out_size, local_model):
        super(LocalModel, self).__init__()
        W = chainer.initializers.HeNormal(1 / np.sqrt(1 / 2), dtype=np.float32)
        bias = chainer.initializers.Zero(dtype=np.float32)
        with self.init_scope():
            self.local_model = local_model
            self.out = L.Linear(out_size=out_size, in_size=None,
                                initial_bias=bias, initialW=W)

    def __call__(self, vertex, edge, adj, length):
        vertex = self.local_model(vertex, adj, edge)[0]
        local_feature = F.vstack([v[:l] for v, l in zip(vertex, length)])
        local_out = self.out(local_feature)
        return local_out


class Model(chainer.Chain):
    def __init__(self, local_model, global_model, stem_model):
        super(Model, self).__init__()
        with self.init_scope():
            self.stem_model = stem_model
            self.local_model = local_model
            self.global_model = global_model

    def predict(self, vertex, edge, adj, length, local=True):
        with function.no_backprop_mode(), chainer.using_config('train', False):
            if local:
                vertex = self.stem_model(vertex, adj, edge)[0]
                local_score = self.local_model(vertex=vertex, edge=edge,
                                               adj=adj, length=length)
                return local_score
            else:
                vertex = self.stem_model(vertex, adj, edge)[0]
                global_score = self.global_model(vertex=vertex, edge=edge,
                                                 adj=adj, length=length)
                return global_score

    def __call__(self, vertex, edge, adj, length):
        vertex = self.stem_model(vertex, adj, edge)[0]
        local_score = self.local_model(vertex=vertex, edge=edge, adj=adj,
                                       length=length)
        global_score = self.global_model(vertex=vertex, edge=edge, adj=adj,
                                         length=length)
        return local_score, global_score


class Classifier(link.Chain):
    def __init__(self, predictor, local_loss_func, global_loss_func, config):
        super(Classifier, self).__init__()
        self.local_loss_func = local_loss_func
        self.global_loss_func = global_loss_func
        self.config = config
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, vertex, edge, adj, length, local_label, global_label,
                 name):
        local_score, global_score = self.predictor(vertex=vertex, edge=edge,
                                                   adj=adj, length=length)

        # Compute local loss
        loss = 0
        if self.config['local_mode']:
            if self.config['local_type'] == 'Classification':
                local_loss = self.local_loss_func(local_score, local_label)
            else:
                local_loss = self.local_loss_func(F.sigmoid(local_score),
                                                  local_label)
            local_loss.name = 'Local Loss'
            chainer.reporter.report({'local_loss': local_loss}, self)
            loss += local_loss

        if self.config['global_mode']:
            # Compute global loss
            global_loss = self.global_loss_func(Variable(global_label),
                                                global_score)
            global_loss.name = 'Global Loss'
            chainer.reporter.report({'global_loss': global_loss}, self)

            # Sum local loss and global loss
            loss += global_loss
        loss.name = 'Loss'
        chainer.reporter.report({'loss': loss}, self)
        return loss


if __name__ == '__main__':
    import json

    config = json.load(open('data/105/last.json', 'r'))['Config'][0]
    model = _get_model(layers=config['model']['global_readout'], comm=None)
    print(model)
