import argparse
from pathlib import Path
import subprocess

pssm_dir = Path.cwd() / "PSSMs"
homologue_dir = Path.cwd() / "Homologues"

if not pssm_dir.is_dir():
	Path.mkdir(pssm_dir)

if not homologue_dir.is_dir():
	Path.mkdir(homologue_dir)

parser = argparse.ArgumentParser()
parser.add_argument("-db_path", help = "path to database such as pdb", type = str)
parser.add_argument("-seq_dir", help = "path to sequence directory", type = str)
args = parser.parse_args()

if args.db_path:
	database = args.db_path
if args.seq_dir:
	query_dir = Path(args.seq_dir)

for query_file in query_dir.iterdir():

	filename = query_file.stem
	out_homologues_file = Path(homologue_dir) / (filename + "_out_homologues.txt")
	num_iters = "3"
	pssm_output = Path(pssm_dir) / (filename + "_pssm.txt")
	subprocess.run(["psiblast", "-query", str(query_file), "-db", database, "-out", out_homologues_file, "-num_iterations", num_iters, "-out_ascii_pssm", pssm_output])
