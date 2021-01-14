from Bio import SeqIO
import os
import argparse
from pathlib import Path
from split import spliting
from generate_pssm_files import generating_pssm
from pseudo_protein import pseudo_proteins


parser = argparse.ArgumentParser()
parser.add_argument("-seq_file", help = "file containing sequences", type = str)
parser.add_argument("-db_path", help = "path to database such as pdb", type = str)
args = parser.parse_args()

sf_index = (Path(args.seq_file).stem)[-5:]
print("The Superfamily Index:\n",sf_index)
if args.db_path:
	database = args.db_path
if args.seq_file:
	filepath = args.seq_file
filename = Path(filepath).stem

dataseqs = Path.cwd()/filename
if not os.path.exists(dataseqs):
	os.mkdir(dataseqs)
print("The given type of file is:\n",filename)

print("...The splitting into single sequence begins...")
spliting(filepath,dataseqs)
print("...The splitting into single sequences ends...")
print("\n")
pssm_dir = Path.cwd() / (filename + "PSSMs")
homologue_dir = Path.cwd() / (filename + "Homologues")
if not pssm_dir.is_dir() :
	Path.mkdir(pssm_dir)

if not homologue_dir.is_dir() :
	Path.mkdir(homologue_dir)


superfamily_file = filename + 'pseudo_protein_seq.txt'

print("...The PSSM and Homologues generation begins...\n")


generating_pssm(dataseqs,database,pssm_dir,homologue_dir)
print("\n")
print("...The PSSM and Homologues generated...\n")


print("...Pseudo Protein Generation begins...")
pseudo_proteins(superfamily_file,sf_index,pssm_dir)
print("...Pseudo Proteins Generated...")



