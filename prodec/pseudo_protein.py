import numpy as np
import Extract_PsSM
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-pssm_dir", help = "path to directory with PSSM files", type = str)
parser.add_argument("-sf_index", help = "superfamily index", type = str)
args = parser.parse_args()

if not Path(args.pssm_dir).is_dir:
	print("DIRECTORY NOT FOUND")
	print(quit)
	quit()
else:
	query_dir = Path(args.pssm_dir)

if args.sf_index:
	superfamily_file = args.sf_index + '_pos.txt'

aa = ['A','R', 'N', 'D', 'C', 'Q','E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T','W', 'Y', 'V']

with open(superfamily_file, 'a') as dataset:

	for child in query_dir.iterdir():

		pssm = np.asarray(Extract_PsSM.pssm_extraction(child))
		indices = np.argmax(pssm, axis = 1)
		pseudo_protein = []
		for idx in indices:
			pseudo_protein.append(aa[idx])
		dataset.write( "".join(pseudo_protein) + '\n')
