import argparse
from pathlib import Path
from acc_hybrid_svm import calc_de1, calc_de2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-pssm_dir", help = "pssm directory path", type = str)
args = parser.parse_args()

if args.pssm_dir:
    pssm_dir = Path(args.pssm_dir)

for pssm_file in pssm_dir.iterdir():

    de1 = calc_de1(pssm_file)
    de2 = calc_de2(pssm_file)
    #acc = np.append(de2, de1, axis=1)
    print(de2.shape)
    print(np.vstack((de2, de1.reshape((1, 20, 3)))).shape)
