import numpy as np
import pandas as pd
from dataset import GraphDataset, convert


class Dataproc():
    def __init__(self, size, rank, config):
        """
        :param size: mpi size
        :param rank: mpi rank
        :param config: experiment config
        """
        self.config = config
        self.rank = rank
        self.size = size
        np.random.seed(0)
        csv_path = config['csv_path']
        path_data = pd.read_csv(csv_path)
        protein_name_list = set(path_data['target'].unique())

        similar_protein = {'T0356', 'T0456', 'T0483', 'T0292', 'T0494', 'T0597', 'T0291', 'T0637', 'T0392', 'T0738',
                           'T0640', 'T0308', 'T0690', 'T0653', 'T0671', 'T0636', 'T0645', 'T0532', 'T0664', 'T0699',
                           'T0324', 'T0303', 'T0418', 'T0379', 'T0398', 'T0518'}
        protein_name_list = protein_name_list - similar_protein
        # protein_name_list = np.sort(list(protein_name_list))
        protein_name_list = np.random.permutation(list(protein_name_list))
        print(protein_name_list[0], rank, size)
        path_data = path_data.reindex(np.random.permutation(path_data.index)).reset_index(drop=True)
        print(path_data.ix[0], rank, size)
        # Split train and validation

        rate = config['train_rate']
        self.protein_name = {'train': protein_name_list[:int(len(protein_name_list) * rate)],
                             'test': protein_name_list[int(len(protein_name_list) * rate):]}
        train_data = path_data.ix[path_data['target'].isin(self.protein_name['train'])]
        test_data = path_data.ix[path_data['target'].isin(self.protein_name['test'])]
        native_data = train_data[train_data['model'] == 'native.npz']
        other_data = train_data[train_data['model'] != 'native.npz']

        # Random Sampling
        frac = config['data_frac']
        other_data = other_data.groupby('target').apply(lambda x: x.sample(frac=frac, random_state=0))
        # test_data = test_data.groupby('target').apply(lambda x: x.sample(frac=frac))
        train_data = pd.concat([native_data, other_data])

        self.data_dict = {}
        path = self.scatter_path_array(path_data=train_data)
        self.data_dict.update({'train': {'path': path}})
        print(path, rank, size)
        path = self.scatter_path_array(path_data=test_data)
        self.data_dict.update({'test': {'path': path}})
        print(path, rank, size)

    def scatter_path_array(self, path_data):
        """
        :param path_data: Dataframe of path data
        :return: each worker's path list
        """
        path_list = path_data['path'].values
        path_list = path_list[self.rank * (len(path_list) // self.size):(self.rank + 1) * (len(path_list) // self.size)]
        return path_list

    def get_protein_name_dict(self):
        """
        :return: protein name list dictation (key = {train, test})
        """
        return self.protein_name

    def get_dataset(self, key):
        """
        :param key: train or test
        :return:
        """
        dataset = GraphDataset(path=self.data_dict[key]['path'], config=self.config)
        return dataset

    def get_converter(self):
        return lambda batch, device: convert(batch=batch, device=device, local_type=self.config['local_type'])


if __name__ == '__main__':
    import json

    f = open('./data/config.json', 'r')
    config = json.load(f)['Config'][0]
    d = Dataproc(1, 0, config)
    data = d.get_dataset('train')
    print(data.get_example(0))
