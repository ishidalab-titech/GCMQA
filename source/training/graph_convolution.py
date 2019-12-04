import chainer.functions as F
import chainer
import chainer.links as L
from training.graph_link import GraphLinear, GraphBias, GraphNorm, GraphActivation, \
    GraphLayerNorm, GraphDropout, GraphThrough, GraphGroupNorm


class NodeAverage(chainer.Chain):
    def __init__(self, out_size, v_in_size=None, initialW=None, initial_bias=None):
        super(NodeAverage, self).__init__()
        with self.init_scope():
            self.c_linear = GraphLinear(in_size=v_in_size, out_size=out_size, initialW=initialW, nobias=True)
            self.n_linear = GraphLinear(in_size=v_in_size, out_size=out_size, initialW=initialW, nobias=True)
            self.b = L.Bias(shape=out_size, axis=2)

    def __call__(self, vertex, adj, *args):
        c = self.c_linear(vertex)
        n = self.n_linear(vertex)
        num = F.sum(adj, axis=1).data
        num = F.where(num == 0, self.xp.ones_like(num), num)
        num = F.reshape(num, (*num.shape, 1))
        n = F.matmul(adj, n) / num
        out = c + n
        out = self.b(out)
        return out, adj


class NodeEdgeAverage(chainer.Chain):
    def __init__(self, v_in_size=None, e_in_size=None, out_size=None, initialW=None, initial_bias=None):
        super(NodeEdgeAverage, self).__init__()
        with self.init_scope():
            self.c_linear = GraphLinear(in_size=v_in_size, out_size=out_size, initialW=initialW, nobias=True)
            self.n_linear = GraphLinear(in_size=v_in_size, out_size=out_size, initialW=initialW, nobias=True)
            self.e_linear = GraphLinear(in_size=e_in_size, out_size=out_size, initialW=initialW, nobias=True)
            self.b = L.Bias(shape=out_size, axis=2)

    def __call__(self, vertex, adj, edge):
        c = self.c_linear(vertex)
        n = self.n_linear(vertex)
        num = F.sum(adj, axis=1).data
        num = F.where(num == 0, self.xp.ones_like(num), num)
        num = F.reshape(num, (*num.shape, 1))
        n = F.matmul(adj, n) / num
        e = self.e_linear(edge)
        e = F.sum(e, axis=1) / num
        out = c + n + e
        out = self.b(out)
        return out, adj, edge
