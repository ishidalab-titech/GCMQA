# GCMQA

## Requirement
- python==3.6
- Chainer==5.2.0
- Prody==1.10.8
- Biopython==1.72
- Emboss [1]
- Scwrl4 [2]
- Blast+ [3]
- SSpro/Accpro 5 [4]
- Rosetta==2018.33 [5]
- Stride [6]
## Preparation

- Python Environment
```bash
pip install biopython==1.72
pip install scipy==1.1.0
pip install pandas==0.23.1
pip install pyparsing==2.2.0
pip install ProDy==1.10.8
pip install chainer==5.2.0
```
- Edit example_path.sh for your environment


## Usage
```bash
cd your_download_directory/source

# Side-chain optimization by Scwrl4
for file in `ls ../sample/pdb_files/*.pdb`; do  Scwrl4 -i ${file} -o ${file}; done

# Generate PSSM and predicted SS and predicted RSA
python preprocess.py -i ../sample/T0759.fasta -o ../sample/profile -d /your_directory/uniref90/uniref90 -n 4

# Predict Model Quality Score from PDBfiles (using CPU)
python predict.py -d ../sample/pdb_files -f ../sample/T0759.fasta -o ../sample/result -m ../data/pretrained_model.npz -p ../sample/profile

# Predict Model Quality Score from PDBfiles (using GPU, device=0)
python predict.py -d ../sample/pdb_files -f ../sample/T0759.fasta -o ../sample/result -m ../data/pretrained_model.npz -p ../sample/profile -g 0

```
## Sample Output
```text
Generate Graph...
Needleman-Wunsch global alignment of two sequences
Predict...
Input Data Path : ../sample/pdb_files/sample_1.pdb
Model Quality Score : 0.30910
Resid	Resname	CAD Score	lDDT Score
13	VAL	0.29541	0.44377
14	ILE	0.11032	0.14017
15	HIS	0.19361	0.09477
16	PRO	0.34011	0.13474
17	ASP	0.66549	0.13497
18	PRO	0.18304	0.06425
19	GLY	0.34607	0.09066
...
```


## Installation and Usage with Docker

### Instalattion
- Download Rosetta==2018.33, Scwrl4, Accpro5

```bash
docker pull ishidalab/gcmqa ???
```

### Usage
```bash
docker run 
-v /your_directory/rosetta_bin_linux_2018.33.60351_bundle:/data/rosetta
-v /your_directory/SCRATCH-1D_1.2:/data/scratch 
-v /your_directory/Scwrl4/:/data/scwrl4
-v /your_directory/uniref90:/data/uniref90
--name sato_gcmqa  ishidalab/gcmqa 
```

## Reference
[1] : EMBOSS: The European Molecular Biology Open Software Suite (2000) Rice,P. Longden,I. and Bleasby,A. Trends in Genetics 16, (6) pp276--277

[2] : G. G. Krivov, M. V. Shapovalov, and R. L. Dunbrack, Jr. Improved prediction of protein side-chain conformations with SCWRL4. Proteins (2009).

[3] : Lipman DJ, Zhang J, Madden TL, Altschul SF, Schäffer AA, Miller W, et al. Gapped BLAST and PSI-BLAST: a new generation of protein database search programs. Nucleic Acids Res [Internet]. 1997;25(17):3389–402. Available from: https://dx.doi.org/10.1093/nar/25.17.3389

[4] : Magnan CN, Baldi P. SSpro/ACCpro 5: almost perfect prediction of protein secondary structure and relative solvent accessibility using profiles, machine learning and structural similarity. Bioinformatics [Internet]. 2014 Sep 15;30(18):2592–7. Available from: http://www.ncbi.nlm.nih.gov/pubmed/24860169

[5] : Alford RF, Leaver-Fay A, Jeliazkov JR, O’Meara MJ, DiMaio FP, Park H, et al. The Rosetta All-Atom Energy Function for Macromolecular Modeling and Design. J Chem Theory Comput. 2017;13(6):3031–48. 

[6] : Heinig M, Frishman D. STRIDE: A web server for secondary structure assignment from known atomic coordinates of proteins. Nucleic Acids Res. 2004;32(WEB SERVER ISS.):500–2. 