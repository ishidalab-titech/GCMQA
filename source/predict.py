from training.model import build_model
from pathlib import Path
import chainer
import numpy as np
import json, argparse
from graph.make_graph import make_graph
from training.dataset import vertex_to_device_batch, edge_adj_to_device_batch
from chainer.backends import cuda


def load_model(config, model_path):
    model = build_model(config=config, comm=None, predict=True)
    obj_path = 'updater/model:main/predictor/'
    chainer.serializers.load_npz(model_path, model, obj_path)
    return model


def predict(input_path, target_path, pssm_path, predicted_ss_path, predicted_rsa_path, model, device, out_path=None):
    print('Generate Graph...')
    vertex, edge, adj, resname, resid = make_graph(input_path=input_path, target_path=target_path, pssm_path=pssm_path,
                                                   predicted_ss_path=predicted_ss_path,
                                                   predicted_rsa_path=predicted_rsa_path)
    vertex = vertex_to_device_batch(arrays=[vertex], device=device)
    edge, adj, num_array = edge_adj_to_device_batch(edge_list=[edge], adj_list=[adj], device=device)
    batch_indices = np.array([vertex.shape[0]])
    print('Predict...')
    local_score, global_score = model.predict(vertex=vertex, edge=edge, adj=adj, num_array=num_array,
                                              batch_indices=batch_indices)
    local_score, global_score = cuda.to_cpu(local_score.data), cuda.to_cpu(global_score.data).ravel()[0]
    cad, lddt = local_score[:, 0], local_score[:, 1]
    print('Input Data Path : {}'.format(input_path))
    print('Model Quality Score : {:.5f}'.format(global_score))
    print('Resid\tResname\tCAD Score\tlDDT Score')
    for i in range(len(resname)):
        print('{}\t{}\t{:.5f}'.format(resid[i], resname[i], cad[i], lddt[i]))
    if out_path:
        out_dict = {'local_cad': cad, 'local_lddt': lddt, 'global_score': global_score}
        np.savez(out_path, **out_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Convolution MQA')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input_path', '-i', help='Input pdb path')
    parser.add_argument('--input_dir_path', '-d', help='Input pdb directory path')
    parser.add_argument('--fasta_path', '-f', help='Reference FASTA Sequence path')
    parser.add_argument('--preprocess_dir_path', '-p', help='Preprocess Directory path')
    parser.add_argument('--model_path', '-m', help='Pre-trained model path')
    parser.add_argument('--out_dir_path', '-o', help='Output directory path')
    args = parser.parse_args()
    if args.input_dir_path:
        input_path_list = Path(args.input_dir_path).glob('*.pdb')
    else:
        input_path_list = Path(args.input_path)

    model = load_model(json.load(open('../data/config.json', 'r')), model_path=args.model_path)
    if args.gpu >= 0:
        model.to_gpu(device=args.gpu)
    preprocess_dir_path = Path(args.preprocess_dir_path)
    pssm_path = preprocess_dir_path / Path(args.fasta_path).with_suffix('.pssm').name
    predicted_ss_path = preprocess_dir_path / Path(args.fasta_path).with_suffix('.ss').name
    predicted_rsa_path = preprocess_dir_path / Path(args.fasta_path).with_suffix('.acc20').name
    out_dir_path = Path(args.out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    for input_path in input_path_list:
        predict(input_path=input_path, target_path=args.fasta_path, pssm_path=pssm_path,
                predicted_ss_path=predicted_ss_path, predicted_rsa_path=predicted_rsa_path, model=model,
                device=args.gpu, out_path=out_dir_path / input_path.with_suffix('.npz').name)
