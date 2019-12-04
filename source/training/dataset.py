import numpy as np
from chainer.dataset import DatasetMixin
from pathlib import Path
from chainer.dataset import concat_examples
import chainer


class GraphDataset(DatasetMixin):

    def __init__(self, path, config, enc):
        super(GraphDataset, self).__init__()
        self.path = path
        self.config = config
        self.enc = enc
        np.random.seed(0)
        self.data_path = Path(config['data_path'])
        self.label_path = Path(config['label_path'])
        self.vertex_feature = config['vertex_feature']

    def __len__(self):
        return len(self.path)

    def get_data(self, path):

        data_path = self.data_path / path
        label_path = self.label_path / path

        data = np.load(data_path)
        vertex, edge, adj = data['vertex'], data['edge'], data['adj']
        vertex_feature_list = []
        if 'base' in self.config['vertex_feature']:
            vertex_feature_list.extend(np.arange(26))
        if 'profile' in self.config['vertex_feature']:
            vertex_feature_list.extend(np.arange(26, 31))
        if 'rosetta' in self.config['vertex_feature']:
            vertex_feature_list.extend(np.arange(31, 51))

        vertex = vertex[:, np.array(vertex_feature_list)]
        edge_feature_list = []
        if 'resid' in self.config['edge_feature']:
            edge_feature_list.extend(np.arange(3))
        if 'angle' in self.config['edge_feature']:
            edge_feature_list.extend(np.arange(3, 4))
        if 'distance' in self.config['edge_feature']:
            edge_feature_list.extend(np.arange(4, 5))
        if 'rosetta' in self.config['edge_feature']:
            edge_feature_list.extend(np.arange(5, 25))
        edge = edge[:, :, np.array(edge_feature_list)]

        label_dict = np.load(label_path, allow_pickle=True)
        local_label = np.zeros(
            [vertex.shape[0], len(self.config['local_label'])])
        for i, label_name in enumerate(self.config['local_label']):
            local_label[:, i] = label_dict[label_name]
        if self.config['local_type'] == 'Classification':
            local_label = (local_label - self.config['local_threshold']) >= 0
            local_label = local_label.astype(np.int32)
        else:
            local_label = local_label.astype(np.float32)
        global_label = label_dict[self.config['global_label']].reshape([1, 1])
        global_label = global_label.astype(np.float32)
        length = vertex.shape[0]
        name = Path(path).parent.name
        name_id = self.enc.transform([name])
        return vertex, edge, adj, length, local_label, global_label, name_id

    def get_example(self, i):
        path = self.path[i]
        return self.get_data(path)


def _to_device(x, device):
    if device is None:
        return x
    else:
        return device.send(x)


class ListDataset(GraphDataset):
    def __init__(self, path, config):
        super(ListDataset, self).__init__(path=path, config=config)


# vertex, edge, adj, local_label, global_label, batch_indices

@chainer.dataset.converter()
def convert(batch, device, local_type):
    data = [list(x) for x in zip(*batch)]
    vertex, edge, adj = convert_data(*data[:3], device)
    length = _to_device(np.array(data[3]), device)
    local_label = np.concatenate(data[4])
    local_dtype = np.float32 if local_type == 'Regression' else np.int32
    local_label = _to_device(local_label.astype(local_dtype), device)
    global_label = _to_device(np.concatenate(data[5]), device)
    name = np.vstack(data[6])
    return vertex, edge, adj, length, local_label, global_label, name


def convert_data(vertex, edge, adj, device):
    vertex = concat_examples(vertex, device=device, padding=0)
    edge = concat_examples(edge, device=device, padding=0)
    adj = concat_examples(adj, device=device, padding=0)
    return vertex, edge, adj
