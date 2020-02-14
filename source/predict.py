from training.model import build_model
from pathlib import Path
import chainer
import chainer.functions as F
import numpy as np
import json, argparse
from graph.make_graph import make_graph
from chainer.backends import cuda
from training.dataset import convert_data


def load_model(config, model_path):
    model = build_model(config=config, comm=None, predict=True)
    obj_path = 'updater/model:main/predictor/'
    chainer.serializers.load_npz(model_path, model, obj_path)
    return model


def predict(input_path, target_path, pssm_path, predicted_ss_path,
            predicted_rsa_path, lddt_model, cad_model, multi_model, device,
            out_path=None, verbose=False):
    print('Generate Graph...')
    vertex, edge, adj, resname, resid = make_graph(input_path=input_path,
                                                   target_path=target_path,
                                                   pssm_path=pssm_path,
                                                   predicted_ss_path=predicted_ss_path,
                                                   predicted_rsa_path=predicted_rsa_path)
    vertex, edge, adj = [vertex], [edge], [adj]
    vertex, edge, adj = convert_data(vertex, edge, adj, device)
    length = np.array([vertex.shape[1]])
    print('Predict...')
    lddt_score = lddt_model.predict(vertex=vertex, edge=edge, adj=adj,
                                    length=length, local=True)

    cad_score = cad_model.predict(vertex=vertex, edge=edge, adj=adj,
                                  length=length, local=True)
    multi_score = multi_model.predict(vertex=vertex, edge=edge, adj=adj,
                                      length=length, local=False)

    lddt_score = cuda.to_cpu(F.sigmoid(lddt_score).data).ravel()
    cad_score = cuda.to_cpu(F.sigmoid(cad_score).data).ravel()
    multi_score = cuda.to_cpu(multi_score.data).ravel()[0]
    global_score = np.mean(lddt_score + cad_score) / 2
    print('Input Data Path : {}'.format(input_path))
    print('GCMQA_lDDT : {:.5f}'.format(lddt_score.mean()))
    print('GCMQA_CAD : {:.5f}'.format(cad_score.mean()))
    print('GCMQA_ensemble : {:.5f}'.format(global_score))
    print('GCMQA_multi : {:.5f}'.format(multi_score))
    if not verbose:
        print('Resid\tResname\tCAD Score\tlDDT Score')
        for i in range(len(resname)):
            print('{}\t{}\t{:.5f}\t{:.5f}'.format(resid[i], resname[i],
                                                  cad_score[i],
                                                  lddt_score[i]))
        if out_path:
            out_dict = {'local_cad': cad_score, 'local_lddt': lddt_score,
                        'global_score': global_score}
            np.savez(out_path, **out_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Convolution MQA')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input_path', '-i', help='Input pdb path')
    parser.add_argument('--input_dir_path', '-d',
                        help='Input pdb directory path')
    parser.add_argument('--fasta_path', '-f',
                        help='Reference FASTA Sequence path')
    parser.add_argument('--preprocess_dir_path', '-p',
                        help='Preprocess Directory path')
    parser.add_argument('--out_dir_path', '-o', help='Output directory path')
    parser.add_argument('--verbose', '-v', help='Disable to show result',
                        action='store_true')
    args = parser.parse_args()
    if args.input_dir_path:
        input_path_list = Path(args.input_dir_path).glob('*.pdb')
    else:
        input_path_list = Path(args.input_path)

    lddt_model = load_model(json.load(open('../data/lddt_config.json', 'r')),
                            model_path='../data/lddt_model.npz')
    cad_model = load_model(json.load(open('../data/cad_config.json', 'r')),
                           model_path='../data/cad_model.npz')
    multi_model = load_model(json.load(open('../data/multi_config.json', 'r')),
                             model_path='../data/multi_model.npz')
    if args.gpu >= 0:
        lddt_model.to_gpu(device=args.gpu)
        cad_model.to_gpu(device=args.gpu)
        multi_model.to_gpu(device=args.gpu)
    preprocess_dir_path = Path(args.preprocess_dir_path)
    pssm_path = preprocess_dir_path / Path(args.fasta_path).with_suffix(
        '.pssm').name
    predicted_ss_path = preprocess_dir_path / Path(args.fasta_path).with_suffix(
        '.ss').name
    predicted_rsa_path = preprocess_dir_path / Path(
        args.fasta_path).with_suffix('.acc20').name
    out_dir_path = Path(args.out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    for input_path in input_path_list:
        predict(input_path=input_path, target_path=args.fasta_path,
                pssm_path=pssm_path, predicted_ss_path=predicted_ss_path,
                predicted_rsa_path=predicted_rsa_path, lddt_model=lddt_model,
                cad_model=cad_model, multi_model=multi_model, device=args.gpu,
                out_path=out_dir_path / input_path.with_suffix('.npz').name,
                verbose=args.verbose)
