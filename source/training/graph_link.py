import chainer
import chainer.links as L
import chainer.functions as F
from functools import partial


class GraphActivation(chainer.Chain):
    def __init__(self, activation, **args):
        super(GraphActivation, self).__init__()
        with self.init_scope():
            if activation.split('.')[0] == 'F':
                self.activation = partial(eval(activation))
            else:
                self.activation = eval(activation)(**args)

    def __call__(self, x, *args):
        h = x
        s0, s1, s2 = h.shape
        h = F.reshape(h, (s0 * s1, s2))
        h = self.activation(h)
        h = F.reshape(h, x.shape)
        return (h, *args)


class GraphDropout(chainer.Chain):
    def __init__(self, ratio):
        self.ratio = ratio
        super(GraphDropout, self).__init__()

    def forward(self, x, *args):
        h = F.dropout(x, ratio=self.ratio)
        return (h, *args)


class GraphLinear(L.Linear):
    def __call__(self, x):
        h = x
        shape = h.shape
        h = F.reshape(h, (-1, shape[-1]))
        h = super(GraphLinear, self).__call__(h)
        h = F.reshape(h, (*shape[:-1], self.out_size))
        return h


class GraphBias(L.Bias):
    def __call__(self, x):
        h = x
        s0, s1, s2 = h.shape
        h = F.reshape(h, (s0 * s1, s2))
        h = super(GraphBias, self).__call__(h)
        h = F.reshape(h, x.shape)
        return h


class GraphNorm(L.BatchNormalization):
    def __call__(self, x, *args):
        h = x
        s0, s1, s2 = h.shape
        h = F.reshape(h, (s0 * s1, s2))
        h = super(GraphNorm, self).__call__(h)
        h = F.reshape(h, x.shape)
        return (h, *args)


class GraphLayerNorm(L.LayerNormalization):
    def __call__(self, x, *args):
        h = x
        s0, s1, s2 = h.shape
        h = F.reshape(h, (s0 * s1, s2))
        h = super(GraphLayerNorm, self).__call__(h)
        h = F.reshape(h, x.shape)
        return (h, *args)


class GraphThrough(chainer.Chain):
    def __init__(self):
        super(GraphThrough, self).__init__()

    def forward(self, x, *args):
        return (x, *args)


class GraphGroupNorm(L.GroupNormalization):
    def __call__(self, x, *args):
        h = x
        s0, s1, s2 = h.shape
        h = F.reshape(h, (s0 * s1, s2))
        h = super(GraphGroupNorm, self).__call__(h)
        h = F.reshape(h, x.shape)
        return (h, *args)
