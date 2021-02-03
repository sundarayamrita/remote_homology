import numpy as np
from pathlib import Path
import argparse
from normalise_aaindex import normalise  

def get_physico_dist(file_loc, aa_index_path):
    
    with open(file_loc, "r") as f:
        seq = f.readline()

    L = len(seq)
    alpha = 10

    lookup = {
            "A":1, "R":2, "N":3, "D":4, "C":5,
            "Q":6, "E":7, "G":8, "H":9, "I":10,
            "L":11, "K":12, "M":13, "F":14, "P":15,
            "S":16, "T":17, "W":18, "Y":19, "V":20
            }
    
    I = normalise(aa_index_path)
    num_props = I.shape[0]
    I = np.transpose(I)
    #print("shape of I", I.shape)
    PDT=[]

    for mu in range(1,alpha+1):

        p1_idx = np.array([lookup.get(seq[i], 1) - 1 for i in range(L - mu)])
        p2_idx = np.array([lookup.get(seq[i], 1) - 1 for i in range(mu , L)])
        p1 = I[p1_idx]
        p2 = I[p2_idx]
        dp = np.sum(np.square(p1 - p2), axis = 0) / (L - mu)
        #print("dp shape", dp.shape)
        PDT.append(dp)

    PDT = np.asarray(PDT).T
    return PDT


if __name__ == "__main__":
    file_loc = Path.cwd() / "query_1.txt"
    aa_index_path = Path.cwd() / "aaindex_format.txt"
    get_physico_dist(file_loc, aa_index_path)
