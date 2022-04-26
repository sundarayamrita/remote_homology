from Bio import SeqIO
import argparse
from pathlib import Path
import sys
sys.path.append("..")
from prodec.split import splitting, splitting_csv
from prodec.generate_pssm_files import generating_pssm
from accumulate_dt import distance_transforms as DT

parser = argparse.ArgumentParser()
parser.add_argument("-seq_file", help = "file containing sequences", type = str)
parser.add_argument("-db_path", help = "path to database such as pdb", type = str)

args = parser.parse_args()

indexed_files = Path.cwd()/"indexed_files"
if not indexed_files.is_dir():
	Path.mkdir(indexed_files)

sf_index = (Path(args.seq_file).stem)
start_ind = sf_index.find('.')
end_ind = len(sf_index)
sf_index = sf_index[start_ind + 1 : end_ind]

print("The Superfamily Index:\n", sf_index)
if args.db_path:
	database = args.db_path
else:
	database = Path.cwd()/"cdd_delta"
if args.seq_file:
	filepath = args.seq_file
filename = Path(filepath).stem

family_files = Path.cwd()/indexed_files/sf_index
dataseqs = family_files/filename

print("The given type of file is:\n", filename)

print("...The splitting into single sequence begins...")
if Path(filepath).suffix == ".csv":
	splitting_csv(filepath, dataseqs)
else:
	if not dataseqs.is_dir():
		Path.mkdir(dataseqs, parents=True)
	splitting(filepath, dataseqs)
print("...The splitting into single sequences ends...")
print("\n")

pssm_dir = family_files / (filename + "_PSSMs")
homologue_dir = family_files /(filename + "_homologues")

if not pssm_dir.is_dir() :
	Path.mkdir(pssm_dir)

if not homologue_dir.is_dir() :
	Path.mkdir(homologue_dir)

superfamily_file = filename + '_pseudo_protein_seq.txt'

if Path(filepath).suffix == ".csv":
	superfamily_file = filename + '_neg_pseudo_protein_seq.txt'
	superfamily_file = filename + '_pos_pseudo_protein_seq.txt'

print("...The PSSM and Homologues generation begins...\n")

generating_pssm(dataseqs, database, pssm_dir, homologue_dir)

print("\n...The PSSM and Homologues generated...\n")

DT(pssm_dir, dataseqs)