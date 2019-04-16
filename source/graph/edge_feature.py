import numpy as np
import pandas as pd
import tempfile
import subprocess


def get_normal_vector(mol):
    ca_list = mol.select('name CA').getCoords()
    c_list = mol.select('name C').getCoords()
    n_list = mol.select('name N').getCoords()
    c_ca_list = c_list - ca_list
    n_ca_list = n_list - ca_list
    vector = np.array([np.cross(c, n) for c, n in zip(c_ca_list, n_ca_list)])
    return vector


def _calc_angle(vec_i, vec_j):
    return np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))


def get_dihedral_angle(normal_vector, adjacency_matrix):
    matrix = np.zeros([adjacency_matrix.shape[0], adjacency_matrix.shape[0], 1])
    for i, j in zip(np.nonzero(adjacency_matrix)[0], np.nonzero(adjacency_matrix)[1]):
        matrix[i][j][0] = _calc_angle(normal_vector[i], normal_vector[j])
    return matrix


def get_distance_matrix(ca_array, adjacency_matrix):
    matrix = np.zeros([adjacency_matrix.shape[0], adjacency_matrix.shape[0], 1])
    for i, j in zip(np.nonzero(adjacency_matrix)[0], np.nonzero(adjacency_matrix)[1]):
        matrix[i][j][0] = np.linalg.norm(ca_array[i] - ca_array[j])
    return matrix


def get_resid_distance_matrix(adjacency_matrix):
    matrix = np.zeros([adjacency_matrix.shape[0], adjacency_matrix.shape[0], 1])
    for i, j in zip(np.nonzero(adjacency_matrix)[0], np.nonzero(adjacency_matrix)[1]):
        matrix[i][j][0] = np.abs(i - j)
    short = (matrix > 0) & (matrix <= 2)
    medium = (matrix > 2) & (matrix <= 8)
    long = matrix > 8
    matrix = np.dstack((short, medium, long))
    return matrix


def get_bond_matrix(adjacency_matrix):
    matrix = np.zeros([adjacency_matrix.shape[0], adjacency_matrix.shape[0], 1])
    for i, j in zip(np.nonzero(adjacency_matrix)[0], np.nonzero(adjacency_matrix)[1]):
        matrix[i][j][0] = (np.abs(i - j) == 1)
    return matrix


def parse_energy(file_path):
    with open(file_path, 'r') as f:
        score = [i.strip().split() for i in f.readlines()]
        df = pd.DataFrame(score[1:], columns=score[0])
        return df


def get_rosetta_energy_from_path(in_path, target_index):
    exe_name = 'residue_energy_breakdown.static.linuxgccrelease'
    file_name = tempfile.mktemp(suffix='.sc')
    cmd = [exe_name, '-in:file:s', in_path, '-out:file:silent', file_name, '-out:level', '100']
    subprocess.call(cmd)
    df = parse_energy(file_path=file_name)
    df.to_csv(file_name)
    df = pd.read_csv(file_name)
    score_column = ['fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4',
                    'lk_ball_wtd', 'fa_elec', 'pro_close', 'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc',
                    'dslf_fa13', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref', 'rama_prepro', 'total']
    rosetta_column = ['resi1', 'resi2', *score_column]
    length = df['resi1'].max() - df['resi1'].min() + 1
    matrix = np.zeros([length, length, len(score_column)])
    df = df[df['resi2'] != '--'][rosetta_column]
    df['resi2'] = df['resi2'].astype(np.int32)
    rosetta_energy = np.array(df)
    for row in rosetta_energy:
        resi1, resi2 = int(row[0]) - 1, int(row[1]) - 1
        matrix[resi1][resi2] = row[2:]
        matrix[resi2][resi1] = row[2:]
    return matrix[target_index][:, target_index]
