import numpy as np
from chainer import backend
from chainer.backends import cuda
import six
from chainer.dataset import DatasetMixin
from pathlib import Path
from scipy import sparse
from chainer.utils import to_coo


class GraphDataset(DatasetMixin):

    def __init__(self, path, config):
        super(GraphDataset, self).__init__()
        self.path = path
        self.config = config
        np.random.seed(0)
        self.data_path = Path(config['data_path'])
        self.label_path = Path(config['label_path'])
        self.local_type = config['local_type']
        self.local_label = config['local_label']
        self.vertex_feature = config['vertex_feature']

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        return len(self.path)

    def get_data(self, path):

        data_path = self.data_path / path
        label_path = self.label_path / path

        data = np.load(data_path)
        vertex, edge, adj = data['vertex'], data['edge'], data['adj']
        vertex_feature_list = []
        if 'base' in self.vertex_feature:
            vertex_feature_list.extend(np.arange(26))
        if 'profile' in self.vertex_feature:
            vertex_feature_list.extend(np.arange(26, 31))
        if 'rosetta' in self.vertex_feature:
            vertex_feature_list.extend(np.arange(31, 51))
        vertex = vertex[:, np.array(vertex_feature_list)]
        label_dict = np.load(label_path)
        local_label = np.zeros([vertex.shape[0], len(self.config['local_label'])])
        for i, label_name in enumerate(self.config['local_label']):
            local_label[:, i] = label_dict[label_name]
        if self.config['local_type'] == 'Classification':
            local_label = (local_label - self.config['local_threshold']) >= 0
            local_label = local_label.astype(np.int32)
        else:
            local_label = local_label.astype(np.float32)

        global_label = np.zeros([1, len(self.config['global_label'])])
        for i, label_name in enumerate(self.config['global_label']):
            global_label[0][i] = label_dict[label_name]
        global_label = global_label.astype(np.float32)
        return vertex, edge, adj, local_label, global_label, vertex.shape[0]

    def get_example(self, i):
        path = self.path[i]
        return self.get_data(path)


def _to_device(device, x):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def vertex_to_device_batch(arrays, device):
    xp = backend.get_array_module(arrays[0])
    concat = xp.concatenate(arrays, axis=0)
    concat = _to_device(device=device, x=concat)
    return concat


def edge_adj_to_device_batch(edge_list, adj_list, device):
    xp = backend.get_array_module(adj_list[0])
    v_num = xp.array([i.shape[0] for i in adj_list])
    adj = xp.zeros([v_num.sum(), v_num.sum()], dtype=np.float32)
    edge = xp.zeros([v_num.sum(), v_num.sum(), edge_list[0].shape[2]], dtype=np.float32)
    v_num = xp.cumsum(v_num)
    for o, n, a, e in zip(xp.array([0, *v_num[:-1]]), v_num, adj_list, edge_list):
        adj[o:n, o:n] = a
        edge[o:n, o:n, :] = e
    edge = edge.reshape([edge.shape[0] ** 2, edge.shape[2]])
    ones = xp.ones(adj.shape[0], dtype=np.float32)
    num_array = xp.sum(adj, axis=1)
    num_array = xp.where(num_array == 0, ones, num_array)
    num_array = num_array.reshape([num_array.shape[0], 1])
    num_array = _to_device(device=device, x=num_array)
    edge = _to_device(device=device, x=edge)
    adj = _to_device(device=device, x=adj)
    edge, adj = to_coo(edge), to_coo(adj)

    return edge, adj, num_array


# vertex, edge, adj, local_label, global_label, batch_indices
def convert(batch, device, local_type):
    vertex_list, edge_list, adj_list, local_label_list, global_label_list, batch_indices_list = [], [], [], [], [], []
    for v, e, a, l, g, b in batch:
        vertex_list.append(v)
        edge_list.append(e)
        adj_list.append(a)
        local_label_list.extend(l)
        global_label_list.extend(g)
        batch_indices_list.append(b)

    vertex = vertex_to_device_batch(arrays=vertex_list, device=device)
    edge, adj, num_array = edge_adj_to_device_batch(edge_list=edge_list, adj_list=adj_list, device=device)

    local_label_list = np.array(local_label_list, dtype=np.float32) if local_type == 'Regression' else np.array(
        local_label_list, dtype=np.int32)
    local_label = _to_device(x=local_label_list, device=device)
    global_label = _to_device(x=np.array(global_label_list, dtype=np.float32), device=device)
    batch_indices = np.cumsum([x for x in batch_indices_list], dtype=np.int32)
    return vertex, edge, adj, num_array, batch_indices, local_label, global_label
