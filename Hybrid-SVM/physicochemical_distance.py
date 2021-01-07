import numpy as np
from pathlib import Path
import argparse
from normalise_aaindex import normalise  
from collections import defaultdict

def get_physico_dist(file_loc, aa_index_path):
    
    with open(file_loc, "r") as f:
        seq = f.readline()

    L = len(seq)
    alpha = 3

    lookup = {
            "A":1, "R":2, "N":3, "D":4, "C":5,
            "Q":6, "E":7, "G":8, "H":9, "I":10,
            "L":11, "K":12, "M":13, "F":14, "P":15,
            "S":16, "T":17, "W":18, "Y":19, "V":20
            }
    
    I = normalise(aa_index_path)
    I = np.transpose(I)
    PDT = np.zeros((531, alpha))
    val = 0

    for mu in range(1, alpha + 1, 1):
        for j in range(531):
            val = 0
            for i in range(L - mu):
                val1 = lookup.get(seq[i], 1)
                val2 = lookup.get(seq[i + mu], 1)
                val = val + (I[val1 - 1, j] - I[val2 - 1, j]) ** 2
                val = val / (L - mu)
            PDT[j, mu - 1] = val

#    print(PDT.shape)
    return PDT

if __name__ == "__main__":
    file_loc = Path.cwd() / "query_1.txt"
    aa_index_path = Path.cwd() / "aaindex_format.txt"
    get_physico_dist(file_loc, aa_index_path)
