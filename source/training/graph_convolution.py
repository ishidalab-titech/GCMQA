import chainer
import chainer.links as L
import chainermn.links as MNL
from training.graph_convolution_link import NodeAverageLink, NodeEdgeAverageLink


class NodeAverage(chainer.Chain):
    def __init__(self, v_in_size=None, out_size=None, nobias=False, initialW=None, initial_bias=None, comm=None,
                 activation=None, batch_norm=None, predict=False, residual=False):
        super(NodeAverage, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual
        with self.init_scope():
            self.gc_model = NodeAverageLink(v_in_size=v_in_size, out_size=out_size,
                                            nobias=nobias, initialW=initialW, initial_bias=initial_bias)
            if batch_norm:
                self.bn = L.BatchNormalization(size=out_size)

    def __call__(self, vertex, edge, adj, num_array):
        v, e, a, j = self.gc_model(vertex=vertex, edge=edge, adj=adj, num_array=num_array)
        if self.batch_norm:
            v = self.bn(v)
        if self.activation:
            v = self.activation(v)
        if self.residual:
            v += vertex
        return v, e, a, j


class NodeEdgeAverage(chainer.Chain):
    def __init__(self, v_in_size=None, e_in_size=None, out_size=None, nobias=False, initialW=None, initial_bias=None,
                 comm=None, activation=None, batch_norm=None, predict=False, residual=False):
        super(NodeEdgeAverage, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual
        with self.init_scope():
            self.gc_model = NodeEdgeAverageLink(v_in_size=v_in_size, e_in_size=e_in_size, out_size=out_size,
                                                nobias=nobias, initialW=initialW, initial_bias=initial_bias)
            if batch_norm:
                if predict:
                    self.bn = L.BatchNormalization(size=out_size)
                else:
                    self.bn = MNL.MultiNodeBatchNormalization(size=out_size, comm=comm)

    def __call__(self, vertex, edge, adj, num_array):
        v, e, a, j = self.gc_model(vertex=vertex, edge=edge, adj=adj, num_array=num_array)
        if self.batch_norm:
            v = self.bn(v)
        if self.activation:
            v = self.activation(v)
        if self.residual:
            v += vertex
        return v, e, a, j
