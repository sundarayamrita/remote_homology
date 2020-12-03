#import os
import argparse
from pathlib import Path
import subprocess

query_dir = Path.cwd() / "Strain_plus"
pssm_dir = Path.cwd() / "PSSMs"
homologue_dir = Path.cwd() / "Homologues"

#if not os.path.exists(pssm_dir):
if not pssm_dir.is_dir():
	Path.mkdir(pssm_dir)
#if not os.path.exists(homologue_dir):
if not homologue_dir.is_dir():
	Path.mkdir(homologue_dir)

parser = argparse.ArgumentParser()
parser.add_argument("-db_path", help = "path to database such as pdb", type = str)

args = parser.parse_args()
if args.db_path:
	database = args.db_path
#print(database)
#print(Path.cwd())
for query_file in query_dir.iterdir():

	filename = query_file.stem
	out_homologues_file = Path(homologue_dir) / (filename + "_out_homologues.txt")
	num_iters = "3"
	pssm_output = Path(pssm_dir) / (filename + "_pssm.txt")
	#blast_cmd = "psiblast -query " + str(query_file) + " -db " + database + " -out " + out_homologues_file + " -num_iterations " + num_iters + " -out_ascii_pssm " + pssm_output
	#os.system(blast_cmd)
	subprocess.run(["psiblast", "-query", str(query_file), "-db", database, "-out", out_homologues_file, "-num_iterations", num_iters, "-out_ascii_pssm", pssm_output])


