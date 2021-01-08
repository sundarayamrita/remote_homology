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
    physico_props = I.shape[0]
    I = np.transpose(I)
#    print("shape of I", I.shape)
    PDT = np.zeros((physico_props, alpha))
    val = 0

    for mu in range(1, alpha + 1):
        for j in range(physico_props):
# AT TS WE WERE TOLD TO AVOID VARIABLES THAT ARE TO BE USED ONLY IN THE NEXT LINE SUCH AS val
# AND IF YOU CANNOT AVOID USING IT THEN GIVE IT A GOOD NAME
#            val = 0
#            for i in range(L - mu):
#                val1 = lookup.get(seq[i], 1)
#                val2 = lookup.get(seq[i + mu], 1)
#                val = val + (I[val1 - 1, j] - I[val2 - 1, j]) ** 2
#                val = val / (L - mu)
             p1_idx = np.array([lookup.get(seq[i], 1) - 1 for i in range(L - mu)])
             p2_idx = np.array([lookup.get(seq[i], 1) - 1 for i in range(mu , L)])

#         j = np.array(range(physico_props))
#         print(j.shape)
#             print(p1_idx.shape)
             P1 = I[p1_idx, j]
             P2 = I[p2_idx, j]
             dp = np.sum((P1 - P2) / (L - mu))
             PDT[j, mu - 1] = val

#    print(PDT.shape)
    return PDT

if __name__ == "__main__":
    file_loc = Path.cwd() / "query_1.txt"
    aa_index_path = Path.cwd() / "aaindex_format.txt"
    get_physico_dist(file_loc, aa_index_path)
