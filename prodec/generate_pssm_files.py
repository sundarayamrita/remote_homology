import argparse
from pathlib import Path
import subprocess
import os
# pssm_dir = Path.cwd() / "PSSMs"
# homologue_dir = Path.cwd() / "Homologues"

# if not pssm_dir.is_dir():
# 	Path.mkdir(pssm_dir)

# if not homologue_dir.is_dir():
# 	Path.mkdir(homologue_dir)

# parser = argparse.ArgumentParser()
# parser.add_argument("-db_path", help = "path to database such as pdb", type = str)
# parser.add_argument("-seq_dir", help = "path to sequence directory", type = str)
# args = parser.parse_args()

# if args.db_path:
# 	database = args.db_path
# if args.seq_dir:
# 	query_dir = Path(args.seq_dir)

# for query_file in query_dir.iterdir():

# 	filename = query_file.stem
# 	out_homologues_file = Path(homologue_dir) / (filename + "_out_homologues.txt")
# 	out_homologues_file=str(out_homologues_file)
# 	num_iters = "3"
# 	pssm_output = Path(pssm_dir) / (filename + "_pssm.txt")
# 	pssm_output=str(pssm_output)
# 	subprocess.run(["psiblast", "-query", str(query_file), "-db", database, "-out", out_homologues_file, "-num_iterations", num_iters, "-out_ascii_pssm", pssm_output])
# 	# blast_cmd = "psiblast -query " + str(filename) + " -db " + database + " -out " + out_homologues_file + " -num_iterations " + num_iters + " -out_ascii_pssm " + pssm_output
# 	# os.system(blast_cmd)

def generate_pssm_files(query_dir, database, pssm_dir, homologue_dir):

	for query_file in query_dir.iterdir():
		filename = query_file.stem
		out_homologues_file = Path(homologue_dir) / (filename + "_out_homologues.txt")
		out_homologues_file=str(out_homologues_file)
		num_iters = "3"
		pssm_output = Path(pssm_dir) / (filename + "_pssm.txt")
		pssm_output=str(pssm_output)
		subprocess.run(["deltablast", "-query", str(query_file), "-db", database, "-out", out_homologues_file, "-num_iterations", num_iters, "-out_ascii_pssm", pssm_output])

def generating_pssm(query_dir, database, pssm_dir, homologue_dir):

	if 'binary' in query_dir.name:
		for gt in ['_neg', '_pos']:
			query_dir = query_dir.parent / (query_dir.name + gt)
			generate_pssm_files(query_dir, database, pssm_dir, homologue_dir)

	else:
		generate_pssm_files(query_dir, database, pssm_dir, homologue_dir)



	# blast_cmd = "psiblast -query " + str(filename) + " -db " + database + " -out " + out_homologues_file + " -num_iterations " + num_iters + " -out_ascii_pssm " + pssm_output
	# os.system(blast_cmd)