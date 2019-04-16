from chainer import Parameter
from chainer import initializers
from chainer import link
import chainer.functions as F
import numpy as np


class NodeAverageLink(link.Link):
    def __init__(self, v_in_size, out_size=None, nobias=False, initialW=None, initial_bias=None, residual=False):
        super(NodeAverageLink, self).__init__()

        if out_size is None:
            v_in_size, out_size = None, v_in_size
        self.out_size = out_size
        self.residual = residual
        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.Wc = Parameter(W_initializer)
            self.Wn = Parameter(W_initializer)
            if v_in_size is not None:
                self._initialize_params_v(v_in_size)
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = Parameter(bias_initializer, out_size)

    def _initialize_params_v(self, v_in_size):
        self.Wc.initialize((v_in_size, self.out_size))
        self.Wn.initialize((v_in_size, self.out_size))

    def __call__(self, vertex, edge, adj, num_array):
        if self.Wc.array is None:
            v_in_size = vertex.shape[1]
            self._initialize_params_v(v_in_size)

        neighbor = F.matmul(vertex, self.Wn)
        neighbor = F.sparse_matmul(adj, neighbor) / num_array
        center = F.matmul(vertex, self.Wc)
        output = center + neighbor
        if self.residual:
            output = vertex + output
        if self.b is not None:
            output += self.b
        return output, edge, adj, num_array


class NodeEdgeAverageLink(link.Link):
    def __init__(self, v_in_size, e_in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super(NodeEdgeAverageLink, self).__init__()

        if out_size is None:
            v_in_size, out_size = None, v_in_size
        self.out_size = out_size
        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.Wc = Parameter(W_initializer)
            self.Wn = Parameter(W_initializer)
            self.We = Parameter(W_initializer)
            if v_in_size is not None:
                self._initialize_params_v(v_in_size)
            if e_in_size is not None:
                self._initialize_params_e(e_in_size)
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = Parameter(bias_initializer, out_size)

    def _initialize_params_v(self, v_in_size):
        self.Wc.initialize((v_in_size, self.out_size))
        self.Wn.initialize((v_in_size, self.out_size))

    def _initialize_params_e(self, e_in_size):
        self.We.initialize((e_in_size, self.out_size))

    def __call__(self, vertex, edge, adj, num_array):
        if self.Wc.array is None:
            v_in_size = vertex.shape[1]
            self._initialize_params_v(v_in_size)
        if self.We.array is None:
            e_in_size = edge.shape[1]
            self._initialize_params_e(e_in_size)
        neighbor = F.matmul(vertex, self.Wn)
        neighbor = F.sparse_matmul(adj, neighbor) / num_array
        center = F.matmul(vertex, self.Wc)
        edge_feature = F.sparse_matmul(edge, self.We)
        length = int(np.sqrt(edge_feature.shape[0]))
        edge_feature = F.reshape(edge_feature, [length, length, edge_feature.shape[1]])
        edge_feature = F.sum(edge_feature, axis=0) / num_array
        output = center + neighbor + edge_feature
        if self.b is not None:
            output += self.b
        return output, edge, adj, num_array
