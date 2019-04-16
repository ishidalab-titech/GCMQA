import numpy as np
import subprocess
import tempfile
from graph import edge_feature, vertex_feature
from prody import parsePDB, LOGGER
from Bio import AlignIO
from functools import reduce

LOGGER.verbosity = 'none'


def _get_neighbor_by_distance(ca_list, center_ca, dist=8):
    dist_list = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=ca_list - center_ca)
    neighbor_array = (dist_list < dist) & (dist_list != 0)
    return neighbor_array


def _get_neighbor_by_num(ca_list, center_ca, num=20):
    dist_list = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=ca_list - center_ca)
    neighbor_array = (dist_list < np.sort(dist_list)[num + 1]) & (dist_list != 0)
    return neighbor_array


def get_adjacency_matrix(ca_list, neighbor_func):
    adjacency_matrix = np.zeros([ca_list.shape[0], ca_list.shape[0]])
    for i, center_ca in enumerate(ca_list):
        adjacency_matrix[i] = neighbor_func(ca_list=ca_list, center_ca=center_ca)
    adjacency_matrix = adjacency_matrix.astype(np.float32)
    return adjacency_matrix


def align_fasta(input_pdb_path, target_fasta_path):
    pdb = parsePDB(input_pdb_path)
    input_fasta_path = tempfile.mktemp(suffix='.fasta')
    f = open(input_fasta_path, 'w')
    f.write('>temp\n')
    if len(pdb.select('name CA').getSequence()) < 25:
        return None, None, None
    else:
        f.write(reduce(lambda a, b: a + b, pdb.select('name CA').getSequence()))
        f.close()
        needle_path = tempfile.mktemp(suffix='.needle')
        cmd = ['needle', '-outfile', needle_path, '-asequence', input_fasta_path, '-bsequence', target_fasta_path,
               '-gapopen', '10', '-gapextend', '0.5']
        subprocess.call(cmd)
        needle_result = list(AlignIO.parse(needle_path, 'emboss'))[0]
        input_seq, target_seq = np.array(list(str(needle_result[0].seq))), np.array(list(str(needle_result[1].seq)))
        input_seq, target_seq = input_seq[np.where(target_seq != '-')], target_seq[np.where(input_seq != '-')]
        input_align_indices = np.where(target_seq != '-')[0]
        target_align_indices = np.where(input_seq != '-')[0]
        align_pdb = pdb.select('resindex ' + reduce(lambda a, b: str(a) + ' ' + str(b), input_align_indices))
        return align_pdb, input_align_indices, target_align_indices


def align_by_resid(input_pdb_path, target_pdb_path):
    input_mol = parsePDB(input_pdb_path)
    target_mol = parsePDB(target_pdb_path)
    target_resid, input_resid = target_mol.select('calpha').getResnums(), input_mol.select('calpha').getResnums()
    target_index = np.where(np.in1d(input_resid, target_resid))[0]
    native_index = np.where(np.in1d(target_resid, input_resid))[0]

    if len(input_mol.select('name CA').getSequence()) < 25 or len(
            target_mol.select('name CA').getSequence()) < 25 or len(target_index) < 25:
        return None, None, None
    input_mol = input_mol.select('resindex ' + reduce(lambda a, b: str(a) + ' ' + str(b), target_index))
    return input_mol, target_index, native_index


def calculate_edge(mol, input_path, target_index, adjacency_matrix):
    normal_vector = edge_feature.get_normal_vector(mol=mol)
    ca_list = mol.select('name CA').getCoords()
    resid_distance = edge_feature.get_resid_distance_matrix(adjacency_matrix=adjacency_matrix)
    dihedral_angle = edge_feature.get_dihedral_angle(normal_vector=normal_vector, adjacency_matrix=adjacency_matrix)
    distance = edge_feature.get_distance_matrix(ca_array=ca_list, adjacency_matrix=adjacency_matrix)

    energy = edge_feature.get_rosetta_energy_from_path(in_path=input_path, target_index=target_index)
    edge_feature_matrix = np.dstack([resid_distance, dihedral_angle, distance, energy])
    edge_feature_matrix = edge_feature_matrix.astype(np.float32)
    return edge_feature_matrix


def calculate_vertex(input_path, mol, pssm_path, predicted_ss_path, predicted_rsa_path,
                     target_index, native_index):
    up_num_list, down_num_list = vertex_feature.get_half_position(mol=mol)
    pssm = vertex_feature.get_pssm(file_path=pssm_path)[native_index]
    observed_rsa = vertex_feature.get_observed_rsa(file_path=input_path)[target_index]
    predcted_rsa = vertex_feature.get_predicted_acc20(file_path=predicted_rsa_path)[native_index]
    observed_ss = vertex_feature.get_observed_ss(file_path=input_path)[target_index]
    predicted_ss = vertex_feature.get_predicted_ss(file_path=predicted_ss_path)[native_index]
    match_ss = np.apply_along_axis(np.all, 1, np.equal(predicted_ss, observed_ss))
    match_ss = np.reshape(match_ss, [match_ss.shape[0], 1])
    rosetta_energy = vertex_feature.get_rosetta_energy_from_pdb(in_path=input_path)[target_index]
    vertex_feature_array = np.hstack(
        [up_num_list, down_num_list, pssm, observed_rsa, predcted_rsa, observed_ss, predicted_ss, match_ss,
         rosetta_energy])
    vertex_feature_array = vertex_feature_array.astype(np.float32)
    return vertex_feature_array


def make_graph(input_path, target_path, pssm_path, predicted_ss_path, predicted_rsa_path,
               neighbor_func=_get_neighbor_by_distance):
    mol, target_index, native_index = align_fasta(input_pdb_path=input_path, target_fasta_path=target_path)
    adj = get_adjacency_matrix(ca_list=mol.select('name CA').getCoords(), neighbor_func=neighbor_func)
    vertex = calculate_vertex(input_path=input_path, mol=mol, pssm_path=pssm_path, target_index=target_index,
                              native_index=native_index, predicted_ss_path=predicted_ss_path,
                              predicted_rsa_path=predicted_rsa_path)
    edge = calculate_edge(mol=mol, target_index=target_index, adjacency_matrix=adj, input_path=input_path)
    return vertex, edge, adj, mol.select('name CA').getResnames(), mol.select('name CA').getResnums()
