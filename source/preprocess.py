import argparse
import subprocess
from pathlib import Path


def run_psiblast(file_path, db_path, out_path, num_threads):
    exec_name = 'psibalst'
    cmd = [exec_name, '-query', file_path, '-db', db_path, '-out_ascii_pssm', out_path, '-num_iterations', '3',
           '-num_threads', num_threads]
    subprocess.call(cmd)


def run_scratch(in_path, out_path, num_threads):
    exe_name = 'run_SCRATCH-1D_predictors.sh'
    cmd = [exe_name, in_path, out_path, num_threads]
    subprocess.call(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--input_path', '-i', help='Reference FASTA Sequence path')
    parser.add_argument('--out_dir_path', '-o', help='Output directory path')
    parser.add_argument('--db_path', '-d', help='Uniref90 DB path for PSIBLAST')
    parser.add_argument('--num_threads', '-n', default='1', type=str, help='Num threads to run blast')
    args = parser.parse_args()
    db_path = args.db_path
    fasta_path = Path(args.input_path)
    out_dir_path = Path(args.out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    print('Run PSI-BLAST...')
    run_psiblast(file_path=fasta_path, db_path=db_path,
                 out_path=out_dir_path / fasta_path.with_suffix('.pssm').name, num_threads=args.num_threads)
    print('Run SSpro / ACCpro 5')
    run_scratch(in_path=fasta_path, out_path=out_dir_path / fasta_path.with_suffix('').name,
                num_threads=args.num_threads)
