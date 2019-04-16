import numpy as np
import pandas as pd
import subprocess
import tempfile


def _half_position(ca_list, center_ca, center_cb):
    """

    Args:
        ca_list: All CA coords
        center_ca: CA coord
        center_cb: CB coord

    Returns: one half sphere exposure

    """
    up_num, down_num = 0, 0
    for ca in ca_list:
        dist = np.linalg.norm(ca - center_ca)
        if dist < 8 and dist != 0:
            if np.dot(ca - center_ca, center_cb - center_ca) >= 0:
                up_num += 1
            else:
                down_num += 1
    return up_num, down_num


def _rot_axis(theta, vector):
    vector = vector / np.linalg.norm(vector)
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    x, y, z = vector
    rot = np.zeros((3, 3))
    # 1st row
    rot[0, 0] = t * x * x + c
    rot[0, 1] = t * x * y - s * z
    rot[0, 2] = t * x * z + s * y
    # 2nd row
    rot[1, 0] = t * x * y + s * z
    rot[1, 1] = t * y * y + c
    rot[1, 2] = t * y * z - s * x
    # 3rd row
    rot[2, 0] = t * x * z - s * y
    rot[2, 1] = t * y * z + s * x
    rot[2, 2] = t * z * z + c
    return rot


def _get_pseudo_cb(n, ca, c):
    n = n - ca
    c = c - ca
    rot = _rot_axis(-np.pi * 120.0 / 180.0, c)
    cb = np.dot(rot, n) + ca
    return cb


def get_all_cb(mol):
    """
    calculate CB (include pseudo for GLY)
    Args:
        mol: protein object (prody.atoms)

    Returns: CB coords list

    """
    cb_list = mol.select('name CB').getCoords()
    resname_list = mol.select('name CA').getResnames()
    pseudo_cb = []
    if mol.select('name N and resname GLY') is not None:
        gly_n = mol.select('name N and resname GLY').getCoords()
        gly_ca = mol.select('name CA and resname GLY').getCoords()
        gly_c = mol.select('name C and resname GLY').getCoords()
        for n, ca, c in zip(gly_n, gly_ca, gly_c):
            pseudo_cb.append(_get_pseudo_cb(n, ca, c))
        pseudo_cb = np.array(pseudo_cb)
        cb_index = np.where(resname_list == 'GLY')[0]
        not_cb_index = np.where(resname_list != 'GLY')[0]
        tmp = np.zeros([resname_list.shape[0], 3])
        tmp[not_cb_index] = cb_list
        tmp[cb_index] = pseudo_cb
    else:
        tmp = cb_list
    return tmp


def get_half_position(mol):
    ca_list = mol.select('name CA').getCoords()
    cb_list = get_all_cb(mol)
    up_num_list, down_num_list = [], []
    for center_ca, center_cb in zip(ca_list, cb_list):
        u, d = _half_position(ca_list=ca_list, center_ca=center_ca, center_cb=center_cb)
        up_num_list.append(u)
        down_num_list.append(d)
    up_num_list = np.array(up_num_list)
    down_num_list = np.array(down_num_list)
    up_num_list = np.reshape(up_num_list, [up_num_list.shape[0], 1])
    down_num_list = np.reshape(down_num_list, [down_num_list.shape[0], 1])
    return up_num_list, down_num_list


def get_stride(file_path):
    ss_7_dict = {'H': 0, 'G': 1, 'I': 2, 'E': 3, 'B': 4, 'b': 4, 'T': 5, 'C': 6}
    ss_dict = {'H': 0, 'G': 0, 'I': 0, 'E': 1, 'B': 1, 'b': 1, 'T': 2, 'C': 2}
    ret = (subprocess.check_output(['stride', file_path], universal_newlines=True))
    ret = ret.split('\n')
    lines = [list(filter(lambda x: len(x) != 0, line.split(' '))) for line in ret[:-1]]
    lines = list(filter(lambda x: x[0] == 'ASG', lines))
    resname, resid, ss, ss_7, asa = [], [], [], [], []
    for line in lines:
        resname.append(line[1])
        resid.append(int(line[3]))
        ss.append(ss_dict[line[5]])
        ss_7.append(ss_7_dict[line[5]])
        asa.append(float(line[9]))
    asa = np.array(asa)
    asa = np.reshape(asa, [asa.shape[0], 1])
    ss = np.identity(3)[np.array(ss)]
    ss = np.reshape(ss, [ss.shape[0], 3])
    return ss, asa


def get_observed_rsa(file_path):
    ret = (subprocess.check_output(['freesasa', '--format=rsa', file_path], universal_newlines=True))
    lines = ret.split('\n')
    lines = list(filter(lambda x: x[:3] == 'RES', lines))
    lines = [i[23:28] for i in lines]
    rsa = np.array([[-1] if i == '  N/A' else [float(i)] for i in lines])
    return rsa


def get_predicted_acc20(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1].strip().split()
        rsa = np.array(lines, dtype=np.float32)
        rsa = np.reshape(rsa, [rsa.shape[0], 1])
        return rsa


def get_observed_ss(file_path):
    ss_dict = {'H': 0, 'G': 0, 'I': 0, 'E': 1, 'B': 1, 'b': 1, 'T': 2, 'C': 2}
    ret = (subprocess.check_output(['stride', file_path], universal_newlines=True))
    ret = ret.split('\n')
    lines = [line.split() for line in ret[:-1]]
    lines = list(filter(lambda x: x[0] == 'ASG', lines))
    ss = np.array([ss_dict[line[5]] for line in lines], dtype=np.int32)
    ss = np.identity(3, dtype=np.bool)[ss]
    return ss


def get_predicted_ss(file_path):
    ss_dict = {'H': 0, 'G': 0, 'I': 0, 'E': 1, 'B': 1, 'b': 1, 'T': 2, 'C': 2}
    with open(file_path, 'r') as f:
        lines = f.readlines()[1].strip()
        ss = np.array([ss_dict[i] for i in lines], dtype=np.int32)
        ss = np.identity(3, dtype=np.bool)[ss]
        return ss


def parse_energy(file_path):
    with open(file_path, 'r') as f:
        score = [i.strip().split() for i in f.readlines()]
        df = pd.DataFrame(score[1:], columns=score[0])
        return df


def get_rosetta_energy_from_pdb(in_path):
    exe_name = 'per_residue_energies.static.linuxgccrelease'
    file_name = tempfile.mktemp(suffix='.sc')
    cmd = [exe_name, '-in:file:s', in_path, '-out:file:silent', file_name, '-out:level', '100']
    subprocess.call(cmd)
    df = parse_energy(file_path=file_name)
    df.to_csv(file_name)
    df = pd.read_csv(file_name)
    rosetta_column = ['fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec',
                      'pro_close', 'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc', 'dslf_fa13', 'omega',
                      'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref', 'rama_prepro', 'score']
    rosetta_energy = np.array(df[rosetta_column])
    return rosetta_energy


def get_pssm(file_path):
    """

    :param file_path: pssm path
    :return: ndarray of pssm
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = lines[3:]
        lines = [list(filter(lambda x: len(x) != 0, l.split(' ')))[2:22] for l in lines]
        lines = list(filter(lambda x: len(x) == 20, lines))
        pssm = np.array(lines, dtype=np.float32)
        return pssm


amino_dict = {'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
              'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18,
              'TYR': 19}


def get_amino(mol):
    resname = mol.select('name CA').getResnames()
    amino = np.zeros([resname.shape[0], 20])
    for index, r in enumerate(resname):
        amino[index][amino_dict[r]] = 1
    return amino.astype(np.float32)
